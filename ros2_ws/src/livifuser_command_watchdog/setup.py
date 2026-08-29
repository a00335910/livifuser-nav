from setuptools import find_packages, setup

package_name = "livifuser_command_watchdog"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/config",
            [
                "config/episode_manager.yaml",
                "config/release_keyboard.yaml",
                "config/rosbag_qos_overrides_v1.yaml",
                "config/watchdog.yaml",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="LiViFuser-Nav Team",
    maintainer_email="maintainer@example.com",
    description="Fail-safe stamped-intent bridge for data acquisition.",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "command_watchdog = livifuser_command_watchdog.node:main",
            "episode_manager = livifuser_command_watchdog.episode_node:main",
            "keyboard_teleop = livifuser_command_watchdog.keyboard_node:main",
            "release_keyboard_teleop = livifuser_command_watchdog.release_keyboard_node:main",
        ],
    },
)
