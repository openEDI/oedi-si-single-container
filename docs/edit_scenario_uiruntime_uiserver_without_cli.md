# Running Directly Using Docker/Podman

Single container consists of **build**, **runtime** and **orchestration** tools. If the user is interested in only the **orchestration** portion, this can be accomplished without using the CLI by directly interacting with Docker/Podman. If you are using Podman instead of Docker, replace all docker commands with podman commands i.e. instead of *docker pull* use *podman pull*

## UIRuntime

The UIRuntime images are available [here](https://hub.docker.com/r/openenergydatainitiative/uiruntime)

You would need to run the following command to get the latest version of the image,

    docker pull openenergydatainitiative/uiruntime:latest


## UIServer

The UIServer images are available [here](https://hub.docker.com/r/openenergydatainitiative/uiserver)

    docker pull openenergydatainitiative/uiserver:latest


## Networking

Containers provide security through namespace isolation. We create a user defined bridge network for different modes (rootful/rootless) of Docker/Podman to work without issues. Here we are creating a subnet at the private IP space 172.20.0.0/24 i.e. 172.20.0.0 - 172.20.0.255. This allows us to assign the network and IP to the containers during runtime.

    docker network create --gateway 172.20.0.1 --subnet 172.20.0.0/24 oedisi_local_network

P.S. You will need to do this only once.

## Run the Container

uiruntime

    docker run --rm --name=uiruntime --net=oedisi_local_network --ip=172.20.0.2 -p 12500:12500 openenergydatainitiative/uiruntime:latest

uiserver

    docker run --rm --name=uiserver --net=oedisi_local_network --ip=172.20.0.3 -p 8080:80 openenergydatainitiative/uiserver:latest

## Stop the Container

uiruntime

    docker stop -t=0 uiruntime

uiserver

    docker stop -t=0 uiserver

## Loading the webpage

Open your browser and go to the following address,

For edit scenario,

    localhost:8080

For analysis (viewing logs and visualization)

    localhost:8080/analysis

If localhost does not resolve as expected, then replace it with 127.0.0.1


## Examples

You can find sample configurations at the following folder,

    examples/edit_scenario/

## Troubleshooting

### Conflict, container name already in use

This implies that a previously run container is either still running or stopped but not removed. The following command will list running containers,

    docker ps

If **uiruntime** and/or **uiserver** is in the list, then you can use,

    docker stop -t=0 uiruntime
    docker stop -t=0 uiserver

The **ps** command with **-a** flag will list all containers, even the stopped ones. 

    docker ps -a

If **uiruntime** and/or **uiserver** is in the list, then you can use,

    docker container rm uiruntime
    docker container rm uiserver
