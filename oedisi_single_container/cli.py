import os
import sys
import uuid
import re
import json
from io import StringIO

import pandas as pd
import click
from .constants import BASE_DIR, COPY_CMD, IS_WINDOWS
from .build import build_for_real


DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "runner", "cli_default_config.json")
DEFAULT_CONFIG = json.load(open(DEFAULT_CONFIG_PATH))


def convert_windows_to_linux_path(windows_path):
    # Replace backslashes with forward slashes
    linux_path = windows_path.replace("\\", "/")

    # Convert drive letter to Linux-style path
    if linux_path[1] == ":":
        drive_letter = linux_path[0].lower()
        linux_path = f"/mnt/{drive_letter}" + linux_path[2:]

    return linux_path


def run_command(cmd):
    print(cmd)
    return os.system(cmd)


@click.group()
def main():
    pass


@main.command()
@click.option(
    "-p", "--project_dir_path", required=True, help="Path to project directory"
)
@click.option("-c", "--config", required=True, help="Path to config file")
@click.option(
    "-r",
    "--run_as_admin",
    required=False,
    default=False,
    help="Should docker be run as root",
)
@click.option(
    "-t",
    "--tag",
    required=False,
    default=DEFAULT_CONFIG["tag"],
    help="Should docker be run as root",
)
@click.option(
    "--podman",
    required=False,
    default=DEFAULT_CONFIG["podman"],
    help="Use podman instead of docker",
)
@click.option(
    "-i", "--interactive", required=False, default=False, help="Interactive mode"
)
def run(project_dir_path, config, run_as_admin, tag, podman, interactive):
    """Runs the co-simulation"""
    project_dir_path = os.path.abspath(project_dir_path)
    config = os.path.abspath(config)
    container_engine = "podman" if podman else "docker"

    directive = ""
    if run_as_admin:
        directive += "sudo "

    name = "singlecontainerapp_" + uuid.uuid4().hex

    if IS_WINDOWS:
        windows_volume_mount(BASE_DIR, project_dir_path, config, tag, container_engine)
        directive = (
            f"{container_engine} run --name {name} "
            + f"-v oedisisc_runtime:/home/runtime "
        )

        if interactive:
            directive += "-it --entrypoint /bin/bash "

        directive += f"{tag}"
        run_command(directive)

        # copy output
        run_command(
            f"wsl -d podman-machine-default -u user enterns podman cp  {name}:/home/output "
            + f'"{convert_windows_to_linux_path(project_dir_path)}"'
        )
        run_command(f"{container_engine} rm {name}")
    else:
        directive += (
            f'{container_engine} run --rm --name {name} -v {os.path.join(BASE_DIR,"runner")}:/home/runtime/runner '
            + f'-v "{config}":/home/runtime/runner/user_config.json '
            + f'-v "{os.path.join(project_dir_path,"user_federates")}":/home/runtime/user_federates '
            + f'-v "{os.path.join(BASE_DIR,"user_interface")}":/home/runtime/user_interface '
            + f'-v "{os.path.join(project_dir_path,"output")}":/home/output '
        )

        if interactive:
            directive += "-it --entrypoint /bin/bash "

        directive += f"{tag}"
        run_command(directive)


def windows_volume_mount(baseDir, project_dir_path, config_path, tag, container_engine):

    # check for missing volumes
    missingVolumes = []
    for entry in ["oedisisc_runtime"]:
        err = run_command(f"{container_engine} volume inspect {entry}")
        if err != 0:
            missingVolumes.append(entry)

    for entry in missingVolumes:
        err = run_command(f"{container_engine} volume create {entry}")

    # mount
    err = run_command(
        f"{container_engine} container create --name oedisisc_dummy -v oedisisc_runtime:/home/runtime {tag}"
    )

    # copy
    run_command(
        f'robocopy "{os.path.join(baseDir,"runner")}" "{os.path.join(project_dir_path,"runner")}" *.*'
    )
    run_command(
        f'robocopy "{os.path.dirname(config_path)}" "{os.path.join(project_dir_path,"runner")}" '
        + config_path.split("\\")[-1]
    )

    if (
        container_engine == "podman"
    ):  # podman cp command doesn't work in Windows. This is a temporary workaround
        err = run_command(
            f'wsl -d podman-machine-default -u user enterns podman cp "{convert_windows_to_linux_path(os.path.join(project_dir_path,"runner"))}" oedisisc_dummy:/home/runtime'
        )
        err = run_command(
            f'wsl -d podman-machine-default -u user enterns podman cp "{convert_windows_to_linux_path(os.path.join(baseDir,"user_interface"))}" oedisisc_dummy:/home/runtime'
        )
        err = run_command(
            f'wsl -d podman-machine-default -u user enterns podman cp "{convert_windows_to_linux_path(os.path.join(baseDir,"user_federates"))}" oedisisc_dummy:/home/runtime'
        )
    else:
        err = run_command(
            f'{container_engine} cp "{os.path.join(project_dir_path,"runner")}" oedisisc_dummy:/home/runtime'
        )
        err = run_command(
            f'{container_engine} cp "{os.path.join(baseDir,"user_interface")}" oedisisc_dummy:/home/runtime'
        )
        err = run_command(
            f'{container_engine} cp "{os.path.join(project_dir_path,"user_federates")}" oedisisc_dummy:/home/runtime'
        )

    # delete
    err = run_command(f"{container_engine} rm oedisisc_dummy")


@main.command(name="init")
@click.option("-p", "--project_dir_path", required=True, help="Path to template folder")
def init(project_dir_path):
    """Initializes a new project"""
    if IS_WINDOWS:
        err = run_command(
            f'mkdir {os.path.join(project_dir_path,"config")} {os.path.join(project_dir_path,"output")}'
        )
        err = run_command(
            f'{COPY_CMD} {os.path.join(BASE_DIR,"runner")} {os.path.join(project_dir_path,"config")}'
        )
        err = run_command(
            f'{COPY_CMD} {os.path.join(BASE_DIR,"user_federates")} {os.path.join(project_dir_path,"user_federates")} /MIR'
        )
    else:
        err = run_command(
            f'mkdir -p {os.path.join(project_dir_path,"config")} {os.path.join(project_dir_path,"output")}'
        )
        assert err == 0, f"creating project directory resulted in error:{err}"
        err = run_command(
            f'{COPY_CMD} {os.path.join(BASE_DIR,"runner","user_config.json")} {os.path.join(project_dir_path,"config")}'
        )
        assert err == 0, f"Copying config resulted in error:{err}"
        err = run_command(
            f'{COPY_CMD} -r {os.path.join(BASE_DIR,"user_federates")} {project_dir_path}'
        )
        assert err == 0, f"Copying user_federates resulted in error:{err}"
        err = run_command(
            f'{COPY_CMD} -r {os.path.join(BASE_DIR,"user_interface")} {project_dir_path}'
        )
        assert err == 0, f"Copying user_interface resulted in error:{err}"


@main.command(name="build")
@click.option(
    "-t", "--tag", required=True, help="Tag to be applied during docker build"
)
@click.option(
    "--nocache",
    required=False,
    type=bool,
    default=False,
    help="apply --no-cache option",
)
@click.option(
    "--podman",
    required=False,
    default=DEFAULT_CONFIG["podman"],
    help="Use podman instead of docker",
)
def build(tag, nocache, podman):
    """Builds a new Docker/Podman Image"""
    build_for_real(tag, nocache=nocache, podman=podman)


@main.command(name="stop")
@click.option(
    "--podman",
    required=False,
    default=DEFAULT_CONFIG["podman"],
    help="Use podman instead of docker",
)
def stop(podman):
    """Stops all running instances of singlecontainerapp_* containers"""
    containerEngine = "podman" if podman else "docker"

    thisFile = f"temp_{uuid.uuid4().hex}.txt"
    run_command(f"{containerEngine} ps --filter name=singlecontainerapp > {thisFile}")
    fpath = os.path.join(os.getcwd(), thisFile)
    with open(fpath) as f:
        data = f.read()

    data = re.sub(r"[ ]{3,}", ",", data)
    data = data.splitlines()
    col = data[0].split(",")
    ind = col.index("CONTAINER ID")
    os.remove(f"{thisFile}")

    ids = []
    for thisLine in data[1::]:
        ids.append(thisLine.split(",")[ind])

    for entry in ids:
        run_command(f"{containerEngine} stop -t 0 {entry}")


@main.command(name="set_default")
@click.option(
    "-p",
    "--python_cmd",
    required=False,
    default=DEFAULT_CONFIG["python_cmd"],
    help="Python command to use i.e. python or python3",
)
@click.option(
    "--podman",
    required=False,
    default=DEFAULT_CONFIG["podman"],
    help="Use podman instead of docker",
)
@click.option(
    "-t",
    "--tag",
    required=False,
    default=DEFAULT_CONFIG["tag"],
    help="Tag to be applied during docker build",
)
def set_default(python_cmd, podman, tag):
    """Set default settings"""
    default_config = json.load(open(DEFAULT_CONFIG_PATH))
    default_config["python_cmd"] = python_cmd
    default_config["podman"] = podman
    default_config["tag"] = tag
    json.dump(default_config, open(DEFAULT_CONFIG_PATH, "w"), indent=3)


@main.command(name="get_default")
def get_default():
    """Get default settings"""
    default_config = json.load(open(DEFAULT_CONFIG_PATH))
    print(json.dumps(default_config, indent=3))


@main.command(name="list_tags")
@click.option(
    "--podman",
    required=False,
    default=DEFAULT_CONFIG["podman"],
    help="Use podman instead of docker",
)
def list_tags(podman):
    """Lists available image tags"""
    container_engine = "podman" if podman else "docker"
    name = "singlecontainerapp"
    run_command(f"{container_engine} images > temp.csv")
    with open("temp.csv") as f:
        data = f.read()
    os.remove("temp.csv")

    data = re.sub(r"[ ]{3,}", ",", data)

    df = pd.read_csv(StringIO(data))
    df[df.REPOSITORY == name]
    available_tags = list(set(df[df.REPOSITORY == name].TAG))
    print([f"{name}:{entry}" for entry in available_tags])


if __name__ == "__main__":
    main()
