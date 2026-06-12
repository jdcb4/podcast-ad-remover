# Deploying to Unraid

Since this application is not yet available in the Community Apps store, you will need to install it using the provided XML template.

## Installation Steps

1.  **Download the Template**: Locate `Documentation/unraid/podcast-ad-remover.xml` in this repository.
2.  **Copy to Flash Drive**: Copy this XML file to your Unraid USB flash drive in the following location:
    ```
    /boot/config/plugins/dockerMan/templates-user/
    ```
    *(You can access this via SMB share at `\\YOUR_UNRAID_IP\flash\config\plugins\dockerMan\templates-user`)*
3.  **Add Container**:
    *   Go to the **Docker** tab in Unraid.
    *   Click **Add Container** at the bottom.
    *   Click the **Template** dropdown menu.
    *   Select **podcast-ad-remover** (it should appear under "User Templates").
4.  **Configuration**:
    *   The template comes pre-configured to use the Docker Hub image `jdcb4/podcast-ad-remover:latest`.
    *   **Data Volume**: Ensure the `/data` path is mapped to where you want your podcast files stored (e.g., `/mnt/user/appdata/podcast-ad-remover`).
    *   **Port**: the web UI and feed server default to port `8000`.
    *   **Session Secret Key**: set a stable random value before enabling dashboard or feed authentication.
5.  **Apply**: Click **Done** to pull the image and start the container.

## Option 2: Manually Add Container (Advanced)

If you prefer to configure the container manually without the XML template:

1.  Go to **Docker > Add Container**.
2.  **Name**: `podcast-ad-remover`
3.  **Repository**: `jdcb4/podcast-ad-remover:latest`
4.  **Network Type**: Bridge
5.  **WebUI**: `http://[IP]:[PORT:8000]/`
6.  **Port Mapping**:
    *   Click "Add another Path, Port, Variable, Label or Device"
    *   Config Type: Port
    *   Container Port: `8000`
    *   Host Port: `8000` (or your preferred port)
7.  **Volume Mapping**:
    *   Click "Add another Path, Port, Variable, Label or Device"
    *   Config Type: Path
    *   Container Path: `/data`
    *   Host Path: `/mnt/user/appdata/podcast-ad-remover`
8.  **Apply**: Click **Done**.

## Configuration

Once running, access the Web UI at `http://YOUR_UNRAID_IP:8000`.

### API Keys
You can set your AI API keys (Gemini, OpenAI, Anthropic, or OpenRouter) directly in the Web UI under **Admin > AI Settings > Text Analysis**. You do not need to pass them as environment variables during installation, although you can if you prefer.

### Public URL
Set `BASE_URL` to a URL your podcast clients can reach, such as `http://YOUR_UNRAID_IP:8000` for LAN-only installs or your HTTPS reverse-proxy URL for remote access.
