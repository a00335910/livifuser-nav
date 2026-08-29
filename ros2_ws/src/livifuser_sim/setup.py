from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = "livifuser_sim"


def recursive_data_files(root: str) -> list[tuple[str, list[str]]]:
    grouped: dict[str, list[str]] = {}
    for raw_path in glob(f"{root}/**/*", recursive=True):
        path = Path(raw_path)
        if not path.is_file():
            continue
        destination = f"share/{package_name}/{path.parent.as_posix()}"
        grouped.setdefault(destination, []).append(path.as_posix())
    return sorted(grouped.items())

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.json")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/worlds", glob("worlds/*.sdf")),
    ]
    + recursive_data_files("models"),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="LiViFuser-Nav Team",
    maintainer_email="maintainer@example.com",
    description="Isolated Gazebo Fortress simulation for LiViFuser-Nav.",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "analytic_lidar = livifuser_sim.analytic_lidar_node:main",
            "contract_node = livifuser_sim.contract_node:main",
            "lidar_policy = livifuser_sim.lidar_policy_node:main",
            "nav2_probe = livifuser_sim.nav2_probe_node:main",
            "privileged_expert = livifuser_sim.privileged_expert_node:main",
            "reactive_expert = livifuser_sim.reactive_expert_node:main",
            "world_pose = livifuser_sim.world_pose_node:main",
        ],
    },
)
