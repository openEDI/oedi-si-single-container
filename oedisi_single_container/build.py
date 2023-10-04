import os
import shutil
from .constants import BASE_DIR, IS_WINDOWS


PREFERRED_BUILD_ORDER = ["pnnl_dsse", "datapreprocessor", "dopf_ornl"]
BUILD_DIR = os.path.join(BASE_DIR, "build")


def fix_white_space(x):
    return '"' + x + '"' if len(x.split(" ")) > 1 else x


def get_docker_info(entry, tmp_dir):
    entry_folder = os.path.join(BUILD_DIR, entry)
    with open(os.path.join(BUILD_DIR, entry, "Dockerfile")) as f:
        new_dockerfile_statements = f.read()

    if "copy_statements.txt" in os.listdir(entry_folder):
        with open(os.path.join(entry_folder, "copy_statements.txt")) as f:
            copy_statements = f.read()
    else:
        copy_statements = ""

    other_docker_contents = set(os.listdir(entry_folder)).difference(
        ["Dockerfile", "copy_statements.txt"]
    )
    if other_docker_contents:
        for filename in other_docker_contents:
            print(
                f"Copying from {os.path.join(entry_folder, filename)} to {os.path.join(tmp_dir, filename)}"
            )
            if os.path.isdir(os.path.join(entry_folder, filename)):
                shutil.copytree(
                    os.path.join(entry_folder, filename),
                    os.path.join(tmp_dir, filename),
                    dirs_exist_ok=True,
                )
            else:
                shutil.copy(os.path.join(entry_folder, filename), tmp_dir)

    return new_dockerfile_statements, copy_statements


def build_for_real(tag, nocache=False, podman=False):
    engine = "podman" if podman else "docker"

    nocache_str = "--no-cache" if nocache else ""

    tmp_dir = os.path.join(BASE_DIR, "tmp")
    if not os.path.exists(tmp_dir):
        os.system(f"mkdir {tmp_dir}")
    if ".gitignore" in os.listdir(tmp_dir):  # ensure that this is the correct directory
        with open(os.path.join(BASE_DIR, "tmp", ".gitignore")) as f:
            temp_data = f.read()

        shutil.rmtree(tmp_dir)
        os.system(f"mkdir {fix_white_space(tmp_dir)}")

        with open(os.path.join(BASE_DIR, "tmp", ".gitignore"), "w") as f:
            f.write(temp_data)

    new_dockerfile, copy_statements = get_docker_info("oedisi", tmp_dir)

    other_docker_items = set(os.listdir(BUILD_DIR)).difference(["oedisi"])
    if set(other_docker_items) == set(PREFERRED_BUILD_ORDER):
        other_docker_items = PREFERRED_BUILD_ORDER
    else:
        other_docker_items = list(other_docker_items)

    for entry in other_docker_items:
        new_dockerfile_statements, new_copy_statements = get_docker_info(entry, tmp_dir)
        new_dockerfile += new_dockerfile_statements
        copy_statements += new_copy_statements

    with open(os.path.join(tmp_dir, "Dockerfile"), "w") as f:
        if IS_WINDOWS:
            new_dockerfile += "\nRUN apt install -y dos2unix"
            f.write(
                new_dockerfile
                + "\n"
                + copy_statements
                + "\nENTRYPOINT dos2unix /home/runtime/runner/run.sh && "
                + "/home/runtime/runner/run.sh"
            )
        else:
            f.write(
                new_dockerfile
                + "\n"
                + copy_statements
                + "\nENTRYPOINT /home/runtime/runner/run.sh"
            )

    # build
    build_command = (
        f"cd {fix_white_space(tmp_dir)} && {engine} build {nocache_str} -t {tag} ."
    )
    print("{build_command}")
    os.system(build_command)
