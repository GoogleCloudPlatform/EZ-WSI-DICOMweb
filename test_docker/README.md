# Testing EZ WSI DicomWeb using Docker

To build the test container run build_docker.sh from root EZ WSI directory.
  Example:

  1. Build docker container named ez_wsi_test.

  ```shell
    sh ./test_docker/build_docker.sh ez_wsi_test
  ```


  2. Run the container interactively.

  ```shell
   docker run -it ez_wsi_test
  ```


  3. Run tests inside the container.

  ```shell
    /test_ez_wsi.sh
  ```


  4. Exit the container.

  ```shell
    /exit
  ```
