# Edit Scenario

Edit scenario allows the user to graphically setup, run and visualize the co-simulation results. This is done through two dedicated Docker Images that are available at DockerHub. The single container **CLI** allows the  

## UIRuntime

The UIRuntime images are available [here](https://hub.docker.com/r/openenergydatainitiative/uiruntime)

## UIServer

The UIServer images are available [here](https://hub.docker.com/r/openenergydatainitiative/uiserver)

## Using the CLI to Pull Docker Images

The following CLI command will pull the latest tag version of the images. If the image is already present and is at the latest version then no action is taken. If the latest version available locally differs from the version available on DockerHub then only the difference layers are pulled.

    oedisisc gui_update

## Using the CLI to List Locally Available UIRuntime and UIServer Images

    oedisisc gui_list_images

## Using the CLI to Start UIRuntime and UIServer

The following command will start both the UIRuntime and UIServer containers. Most of the time the user will only work with two CLI commands, *gui_start* and *gui_stop*.

    oedisisc gui_start

## Using the CLI to Stop UIRuntime and UIServer

The following command will stop both the UIRuntime and UIServer containers. Most of the time the user will only work with two CLI commands, *gui_start* and *gui_stop*.

    oedisisc gui_stop

## Using the CLI to get UIRuntime and UIServer Container Status

To get the status of the UIRuntime and UIServer containers use,

    oedisisc gui_status
