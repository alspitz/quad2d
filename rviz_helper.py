import time

import rospy

from geometry_msgs.msg import Point, TransformStamped
from std_msgs.msg import ColorRGBA
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker

import tf2_ros

class RVIZHelper:
  def __init__(self):
    rospy.init_node("rviz_helper", disable_signals=True)

    ns = "danaus06/motion_manager"
    self.carrot_pub = rospy.Publisher(rospy.names.ns_join(ns, "carrot_vis"), Marker, queue_size=20)
    self.traj_pub = rospy.Publisher(rospy.names.ns_join(ns, "traj_vis"), Marker, queue_size=20)

    time.sleep(1)

    self.broadcaster = tf2_ros.TransformBroadcaster()
    self.transform = TransformStamped()
    self.transform.header.frame_id = "world"
    self.transform.child_frame_id = "danaus06"
    self.transform2 = TransformStamped()
    self.transform2.header.frame_id = "world"
    self.transform2.child_frame_id = "danaus06/base"

    self.carrot_marker = Marker()
    self.carrot_marker.header.frame_id = "world"
    self.carrot_marker.type = Marker.SPHERE
    self.carrot_marker.action = Marker.ADD
    self.carrot_marker.scale.x = 0.05
    self.carrot_marker.scale.y = 0.05
    self.carrot_marker.scale.z = 0.05
    self.carrot_marker.color.a = 1.0
    self.carrot_marker.color.r = 0.929
    self.carrot_marker.color.g = 0.569
    self.carrot_marker.color.b = 0.129
    self.carrot_marker.pose.orientation.w = 1

    self.traj_marker = Marker()
    self.traj_marker.header.frame_id = "world"
    self.traj_marker.type = Marker.LINE_STRIP
    self.traj_marker.action = Marker.ADD
    self.traj_marker.scale.x = 0.03
    self.traj_marker.pose.orientation.w = 1
    self.traj_marker.color.a = 1.0
    self.traj_marker.color.b = 1.0

  def convert_pos(self, x, z):
    return [0, -x, 1.5 + z]

  def publish_robot(self, x, z, angle):
    pos = self.convert_pos(x, z)
    quat = quaternion_from_euler(-angle, 0, 0)

    self.transform.header.stamp = rospy.Time.from_sec(time.time())
    self.transform.transform.translation.x = pos[0]
    self.transform.transform.translation.y = pos[1]
    self.transform.transform.translation.z = pos[2]
    self.transform.transform.rotation.x = quat[0]
    self.transform.transform.rotation.y = quat[1]
    self.transform.transform.rotation.z = quat[2]
    self.transform.transform.rotation.w = quat[3]
    self.transform2.header = self.transform.header
    self.transform2.transform = self.transform.transform

    self.broadcaster.sendTransform(self.transform)
    self.broadcaster.sendTransform(self.transform2)

  def publish_trajectory(self, xs, zs):
    self.traj_marker.points = []
    for x, z in zip(xs, zs):
      point_msg = Point()
      pos = self.convert_pos(x, z)
      point_msg.x = pos[0]
      point_msg.y = pos[1]
      point_msg.z = pos[2]
      self.traj_marker.points.append(point_msg)

    self.traj_pub.publish(self.traj_marker)

  def publish_marker(self, x, z):
    pos = self.convert_pos(x, z)
    self.carrot_marker.pose.position.x = pos[0]
    self.carrot_marker.pose.position.y = pos[1]
    self.carrot_marker.pose.position.z = pos[2]
    self.carrot_pub.publish(self.carrot_marker)

if __name__ == "__main__":
  rviz_helper = RVIZHelper()

  while not rospy.is_shutdown():
    rospy.spin()
