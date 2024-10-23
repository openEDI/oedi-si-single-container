# Single container build process
This is a brief introduction of how the Single container build process works:

1. **Build single docker/podman image**: Here the objective is to combine images from **oedisi, oedisi-example, datapreprocessor, DSSE, and DOPF** applications into a single image. The automatic build process will pull the latest Dockerfiles from all these packages and build the image. The script for this is in **build.py** and specifications for each applications is give in **specification.json**
    ```shell
    oedisisc build -t singlecontainerapp:0.1.0 --nocache false -p python
    ```

    The automated build process will also do the following:
    a. Remove all FROM statements after the first FROM statement.
    b. Modify the WORKDIR and COPY statements from application Dockefile such that they are placed in specific application folders. 
    c. Install Python >=3.11
    
    Alternate build command if the image needs to be rebuild with podman/docker.
    ```shell
    podman build -t singlecontainerapptest:0.1.0 .
    ```
    
2. Initialize the configuration files into a user defined project directory.
    ```shell
    oedisisc init -p "C:\\Users\\splathottam\Box Sync\\GitHub\\oedi-si-single-container\\testproject"
    ```

3. Start the the single container. This will also mount the user defined project directory as well as the runner folder from oedisi-single-container onto the single container.
    ```shell
    oedisisc run -p testproject -c testproject/config/user_config.json --tag singlecontainerapptest:0.1.0 -i true
    ```
    3a. Create configuration files within the container by running the parse_config.py script. This script will create the following file **/home/run/system_runner.json**.
    
    ```shell
    python /home/runtime/runner/parse_config.py
    ```
     3b. Start the HELICS cosimulation using 
    ```shell
    helics run --path /home/run/system_runner.json 
    ```
