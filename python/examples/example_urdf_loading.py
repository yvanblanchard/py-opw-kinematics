"""
Example demonstrating URDF loading functionality
"""

import py_opw_kinematics as opw

# Example 1: Load from URDF string
urdf_xml = """<?xml version="1.0"?>
<robot name="test_robot">
    <joint name="joint1" type="revolute">
        <origin xyz="0 0 0.45" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14"/>
    </joint>
    <joint name="joint2" type="revolute">
        <origin xyz="0.15 0 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-1.57" upper="1.57"/>
    </joint>
    <joint name="joint3" type="revolute">
        <origin xyz="0 0 0.6" rpy="0 -1.5708 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-3.14" upper="3.14"/>
    </joint>
    <joint name="joint4" type="revolute">
        <origin xyz="0 0 0.2" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <limit lower="-3.31" upper="3.31"/>
    </joint>
    <joint name="joint5" type="revolute">
        <origin xyz="0.615 0 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-3.31" upper="3.31"/>
    </joint>
    <joint name="joint6" type="revolute">
        <origin xyz="0.1 0 0" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <limit lower="-6.28" upper="6.28"/>
    </joint>
</robot>
"""

try:
    # Load kinematic model from URDF string
    print("Loading robot parameters from URDF string...")
    kinematic_model = opw.from_urdf_string(urdf_xml)
    print(f"Successfully loaded kinematic model:\n{kinematic_model}")
    
    # Create robot instance
    robot = opw.Robot(kinematic_model, degrees=True)
    print(f"\nCreated robot:\n{robot}")
    
    # Test forward kinematics
    joints = (0, 0, 0, 0, 0, 0)
    pose = robot.forward(joints)
    print("\nForward kinematics at zero position:")
    print(f"Translation: {pose.translation}")
    print(f"Rotation: {pose.rotation.as_euler('xyz', degrees=True)}")
    
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60)
print("Example completed!")
