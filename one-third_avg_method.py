#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node

from sensor_msgs.msg import PointCloud
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

from sklearn.cluster import DBSCAN


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

        cone_points_list = [] #CONTAINS INTENSITY
        j=0
        
        # Z THRESHOLDING:
        for i in point_detarr:
            if i[2] > -0.1629:  
                j = j+1
                cone_points_list.append(i)

        self.get_logger().info(f"Received {len(point_detarr)} points")


        # Checking eligibility for DBSCAN to be applied
        if len(cone_points_list) > 0:
            cone_points = np.array(cone_points_list) #CONTAINS INTENSITY
            clustering = DBSCAN(eps=1, min_samples=2).fit(cone_points[:, :3]) # HERE INTENSITY GETS LOST
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
            
        for val in clusters.values():
            centroid = np.mean(val, axis=0)
            z_min = np.min(val[:, 2])
            z_max = np.max(val[:, 2])
            third_height = (z_max - z_min) / 3.0 
            one_av = 0.0
            count_one = 0
            two_av = 0.0
            count_two = 0
            three_av = 0.0
            count_three = 0
            for point in val:
                if point[2] < third_height + z_min:
                    one_av += point[3]
                    count_one += 1
                elif point[2] < 2 * third_height + z_min:
                    two_av += point[3]
                    count_two += 1
                elif point[2] < 3 * third_height + z_min:
                    three_av += point[3]
                    count_three += 1
            if count_one > 0:
                one_av /= count_one
            if count_two > 0:
                two_av /= count_two
            if count_three > 0:
                three_av /= count_three

            if two_av > one_av and two_av > three_av:
                centroid[3] = 0.0
            elif three_av > two_av and one_av > two_av:
                centroid[3] = 1.0
            else:
                centroid[3] = 0.5
                    
            self.cone_points_cent.append(centroid)
            self.get_logger().info(f"centroid {centroid} ")

        self.get_logger().info(f"Finall {j} points")


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
            marker.color.g = 0.0
            marker.color.b = abs(pt[3]-1.0)
            marker.color.a = 1.0

            marker.lifetime = Duration(sec=0, nanosec=0)
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
