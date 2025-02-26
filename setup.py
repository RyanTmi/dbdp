from setuptools import setup, find_packages


def main() -> None:
    setup(
        name="dbdp",
        version="0.1",
        packages=find_packages(),
        # author="",
        description="",
        install_requires=["torch", "numpy", "tqdm"],
    )


if __name__ == "__main__":
    main()
