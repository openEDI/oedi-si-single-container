# Installing Docker

## Linux

Rootful Docker

	apt install docker.io

Rootless Docker

Details about rootless docker and installation instructions are available at https://docs.docker.com/engine/security/rootless/

## Windows

### Prerequisites
- Windows 10/11 (Pro, Enterprise, or Education) with WSL 2 enabled
- Windows 10 Home users need to install WSL 2
- Virtualization enabled in BIOS
- Admin privileges

### Install Docker on Windows
1. **Download Docker Desktop**  
   [Get Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. **Run the Installer**  
   - Follow the installation wizard
   - Ensure "Use WSL 2 instead of Hyper-V" is selected (if available)
3. **Verify Installation**  
   Open PowerShell or Command Prompt and run:
   ```sh
   docker --version
   docker run hello-world
   ```

### Install Podman on Windows
1. **Download Podman**  
   [Get Podman](https://podman.io/getting-started/installation)
2. **Install Using Winget (Recommended)**  
   ```sh
   winget install --id=RedHat.Podman -e
   ```
3. **Enable WSL Backend (Optional for Full Compatibility)**  
   ```sh
   podman machine init
   podman machine start
   ```
4. **Verify Installation**  
   ```sh
   podman --version
   podman run hello-world
   ```

### Additional Resources
- [Docker Documentation](https://docs.docker.com/desktop/install/windows-install/)
- [Podman Documentation](https://podman.io/)

## MacOS

To be completed.
