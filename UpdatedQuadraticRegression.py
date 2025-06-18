#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
import matplotlib.pyplot as plt

from sensor_msgs.msg import PointCloud
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor


class ConeVisualisation(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            PointCloud,
            'carmaker/pointcloud',
            self.listener_callback,
            10)
        self.subscription
        self.publisher_ = self.create_publisher(MarkerArray, 'viz', 10)
        self.i = 0
        self.cone_points_cent = []
        self.marker_id = 0
        self.blue_correct = 0
        self.yellow_correct = 0
        self.yellow_count = 0
        self.blue_count = 0


    def listener_callback(self, msg):
        
        point_detarr = []
        for i in msg.points:
            point_detarr.append([i.x, i.y,i.z, 0.0])
        j = 0
        for channel in msg.channels:
            if channel.name == "intensity":
                intensities = channel.values
                for i in range(min(len(point_detarr), len(intensities))):
                    point_detarr[i][3] = intensities[i]

        '''cone_points_list = [] #CONTAINS INTENSITY
        j=0
        
        # Z THRESHOLDING:
        for i in point_detarr:
            if i[2] > -0.1579:  
                j = j+1
                cone_points_list.append(i)

        self.get_logger().info(f"Received {len(point_detarr)} points")'''


        points_np = np.array(point_detarr)

        if len(points_np) < 10:
            self.get_logger().warn("Not enough points for RANSAC.")
            return

        # Extract coordinates for RANSAC: X = (x, y), y = z
        X = points_np[:, :2]
        y = points_np[:, 2]

        # Fit RANSAC plane model: z = ax + by + c
        ransac = RANSACRegressor(residual_threshold=0.1, max_trials=100)
        try:
            ransac.fit(X, y)
        except Exception as e:
            self.get_logger().error(f"RANSAC fitting failed: {e}")
            return

        z_pred = ransac.predict(X)
        residuals = np.abs(y - z_pred)

        # Filter out ground points
        non_ground_mask = residuals > 0.02
        filtered_points = points_np[non_ground_mask]
        cone_points_list = filtered_points.tolist()

        if len(cone_points_list) == 0:
            self.get_logger().warn("No points remaining after ground removal.")
            return


        # Checking eligibility for DBSCAN to be applied
        if len(cone_points_list) > 0:
            cone_points = np.array(cone_points_list) #CONTAINS INTENSITY
            clustering = DBSCAN(eps=1, min_samples=7).fit(cone_points[:, :3]) # HERE INTENSITY GETS LOST
            labels = clustering.labels_ 
            self.get_logger().info(f"DBSCAN found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
        else:
            self.get_logger().warn("No points available for clustering.")
            return

        unique_labels = set(labels)

        self.cone_points_cent = []

        clusters = {}  # Dictionary to hold points per cluster
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise
            cluster_points = cone_points[labels == label]
            clusters[label] = cluster_points
            
        con_matrix = []

        
        for val in clusters.values():
            centroid = np.mean(val, axis=0)
            
            dist = np.sqrt(centroid[0]**2 + centroid[1]**2)
            '''if dist > 10.0:
                self.get_logger().warn(f"Skipping cluster at distance {dist} > 20.0")
                continue'''


            val_sorted = sorted(val, key=lambda p: p[2])
            z_vals = np.array([p[2] for p in val_sorted])
            intensities = np.array([p[3] for p in val_sorted])

           



            if len(z_vals) > 3 and centroid[0]>1:
                # Fit a 2nd-degree polynomial to intensity vs z

               

                
                coeffs = np.polyfit(z_vals, intensities, deg=2)
                poly = np.poly1d(coeffs)

                # Generate smooth line for the polynomial curve
                '''z_smooth = np.linspace(z_vals.min(), z_vals.max(), 100)
                intensity_fit = poly(z_smooth)
                # Plotting
                plt.figure(figsize=(6, 4))
                plt.plot(z_vals, intensities, 'bo-', label='Original Intensity')
                plt.plot(z_smooth, intensity_fit, 'r-', linewidth=2, label='Fitted Polynomial')
                plt.title('Cone Intensity vs Z with Polynomial Fit')
                plt.xlabel('Z (height)')
                plt.ylabel('Intensity')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show()'''

                mid_z = (z_vals[0] + z_vals[-1]) / 2
                int_mid = poly(mid_z)
                
                int_start = poly(z_vals[0])
                int_end = poly(z_vals[-1])
                int_quarter = poly((z_vals[0] + z_vals[-1]) / 4)

                if coeffs[0] < 0:  # Check if the polynomial is concave down
                    cone_type = 0.0
                elif coeffs[0] > 0:  # Check if the polynomial is concave up
                    cone_type = 1.0
                else:
                    cone_type = 0.5

                if centroid[1] > 0:
                    true_color = 0.0
                    self.blue_count += 1
                else:
                    true_color = 1.0
                    self.yellow_count += 1
                
                if abs(cone_type - true_color)  < 1e-3:
                    if true_color == 0.0:
                        self.blue_correct += 1
                    else:
                        self.yellow_correct += 1
                
                '''if centroid[0] >5 and centroid[0]<15 and centroid[1] < 0 and cone_type == 0.0:
                    z_smooth = np.linspace(z_vals.min(), z_vals.max(), 100)
                    intensity_fit = poly(z_smooth)
                    # Plotting
                    plt.figure(figsize=(6, 4))
                    plt.plot(z_vals, intensities, 'bo-', label='Original Intensity')
                    plt.plot(z_smooth, intensity_fit, 'r-', linewidth=2, label='Fitted Polynomial')
                    plt.title('Cone Intensity vs Z with Polynomial Fit')
                    plt.xlabel('Z (height)')
                    plt.ylabel('Intensity')
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()'''

            elif centroid[0] < 1:
                if centroid[1] > 0:
                    cone_type = 0.0
                    self.blue_count += 1
                else:
                    cone_type = 1.0
                    self.yellow_count += 1

            else:
                continue
                    


                

            centroid = np.mean(val, axis=0)
            centroid[3] = cone_type 
            con_matrix.append([self.blue_correct, self.yellow_correct, self.blue_count, self.yellow_count])  
            self.get_logger().info(f"Appended to con_matrix: {con_matrix[-1]}")

            self.cone_points_cent.append(centroid)
            self.get_logger().info(f"centroid {centroid} ")

        #self.get_logger().info(f"Finall {j} points")


        marker_arr = MarkerArray()
        self.marker_id = 0

        del_marker = Marker()
        del_marker.action = Marker.DELETEALL
        marker_arr.markers.append(del_marker)

        for pt in self.cone_points_cent:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()

            marker.ns = "basic_shapes"
            marker.id = self.marker_id
            self.marker_id +=1

            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = float(pt[0])
            marker.pose.position.y = float(pt[1])
            marker.pose.position.z = 0.4

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.8

            marker.color.r = pt[3]
            marker.color.g = pt[3]
            marker.color.b = abs(pt[3]-1.0)
            marker.color.a = 1.0

            marker.lifetime = Duration(sec=0, nanosec=00000000)  # 0.5 seconds
            marker_arr.markers.append(marker)
    
        self.publisher_.publish(marker_arr)
        self.get_logger().info("Published cylinders")

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = ConeVisualisation()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
