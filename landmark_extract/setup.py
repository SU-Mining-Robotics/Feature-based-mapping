from setuptools import find_packages, setup
from setuptools import setup
from glob import glob
import os

package_name = 'landmark_extract'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
         # Add message definitions
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ruan',
    maintainer_email='delport121@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "Bezier = landmark_extract.Bezier_curve_extract:main",
            "Path_gt = landmark_extract.Path_test_gt:main",
            "Path_odom = landmark_extract.Path_test_odom:main",
            "Scan_process = landmark_extract.Lidar_scan_processing:main",
            "EKF_SLAM = landmark_extract.EKF_SLAM_NODE:main",
            "Feature_scan_SLAM = landmark_extract.Feature_scan_SLAM:main",
        ],
    },
)
