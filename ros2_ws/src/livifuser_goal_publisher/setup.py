from setuptools import find_packages, setup

package_name = "livifuser_goal_publisher"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", ["config/odom_waypoint.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="LiViFuser-Nav Team",
    maintainer_email="maintainer@example.com",
    description="Validated relative-goal publisher for pilot data collection.",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "odom_waypoint_goal_publisher = livifuser_goal_publisher.odom_node:main",
            "relative_goal_publisher = livifuser_goal_publisher.node:main",
        ],
    },
)
