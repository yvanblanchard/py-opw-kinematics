"""
Example demonstrating URDF loading functionality
"""
import csv
import math
import xml.etree.ElementTree as ET
import py_opw_kinematics as opw
from scipy.spatial.transform import RigidTransform, Rotation

def load_urdf_data():

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

def load_urdf_from_file():
    # Example 2: Load from URDF file
    urdf_file_path = "C:\\YVAN\\CODE\\urdf_viser_visualizer\\urdf2\\ABB_IRB_7600.urdf"  # Replace with your URDF file path

    try:
        # Parse URDF to extract joint names
        tree = ET.parse(urdf_file_path)
        root = tree.getroot()
        joint_names = []
        for joint in root.findall('.//joint[@type="revolute"]'):
            joint_name = joint.get('name')
            if joint_name:
                joint_names.append(joint_name)
        
        print(f"Found {len(joint_names)} revolute joints: {joint_names}")
        
        # Load kinematic model from URDF file
        print(f"Loading robot parameters from URDF file: {urdf_file_path}...")
        kinematic_model = opw.from_urdf_file(urdf_file_path)
        print(f"Successfully loaded kinematic model:\n{kinematic_model}")
        
        # Create robot instance
        robot = opw.Robot(kinematic_model, degrees=False)
        print(f"\nCreated robot:\n{robot}")

        # print robot model parameters
        print("\nRobot model parameters:")
        print(kinematic_model)
        
        # Test IK with target
        target_pose = RigidTransform.from_components(
            rotation=Rotation.from_euler('xyz', [180, 90, 180], degrees=True),
            translation=[2.0, 0.0, 2.7]
        )

        ik_solutions = robot.inverse(target_pose)
        print("\nInverse kinematics solutzions for target pose:")
        print("Target pose:")
        print(f"  Translation: {target_pose.translation}")
        print(f"  Rotation (euler xyz): {target_pose.rotation.as_euler('xyz', degrees=True)}")
        
        # Verify each solution with FK
        print("\n" + "="*60)
        print("Verifying IK solutions with FK:")
        for idx, solution in enumerate(ik_solutions):
            print(f"\nSolution {idx + 1}: {solution}")
            
            # Compute FK for this solution
            fk_pose = robot.forward(solution)
            
            # Compute position error
            position_error = fk_pose.translation - target_pose.translation
            position_error_norm = (position_error[0]**2 + position_error[1]**2 + position_error[2]**2)**0.5
            
            # Compute rotation error (angle between rotations)
            rotation_diff = target_pose.rotation.inv() * fk_pose.rotation
            angle_error = rotation_diff.magnitude()
            
            print(f"  FK Translation: {fk_pose.translation}")
            print(f"  FK Rotation (euler xyz): {fk_pose.rotation.as_euler('xyz', degrees=True)}")
            print(f"  Position error: {position_error_norm:.6e} m")
            print(f"  Rotation error: {angle_error:.6e} rad ({angle_error * 180 / math.pi:.6e} deg)")
            
            if position_error_norm < 1e-6 and angle_error < 1e-6:
                print("  ✓ Solution verified (within tolerance)")
            else:
                print("  ✗ Solution has significant error")

        # Write CSV file with timestamp and robot joints from all IK solutions
        if ik_solutions:
            duration = 0.0
            rail_axis = 0.0  # Replace with actual rail axis value if available
            output_file = 'robot_poses.csv'
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header with actual joint names from URDF
                header = ['timestamp', 'rail'] + joint_names[:6]  # Use first 6 joint names
                writer.writerow(header)
                
                # Write data rows with all IK solutions
                for solution in ik_solutions:
                    writer.writerow([f'{duration:.3f}'] + [f'{rail_axis:.3f}'] + [f'{joint:.6f}' for joint in solution])
                    duration += 1.0
            
            print(f"\nRobot pose saved to {output_file} ({len(ik_solutions)} solutions)")
        else:
            print("\nNo IK solutions found, CSV not created.")

        
    except FileNotFoundError:
        print(f"URDF file not found: {urdf_file_path}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "="*60)
    print("Example completed!")


if __name__ == "__main__":

    #load_urdf_data()

    load_urdf_from_file()
