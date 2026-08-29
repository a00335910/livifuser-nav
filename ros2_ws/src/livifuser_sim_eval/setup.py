from setuptools import setup

package_name = "livifuser_sim_eval"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="LiViFuser-Nav Team",
    maintainer_email="maintainer@example.com",
    description="Closed-loop evaluation entry points for LiViFuser-Nav.",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "constant_arm_runner = livifuser_sim.constant_arm_runner_node:main",
            "live_policy_runner = livifuser_sim.live_policy_runner_node:main",
            "simulation_supervisor = livifuser_sim.simulation_supervisor_node:main",
        ],
    },
)
