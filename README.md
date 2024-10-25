# EZ WSI DicomWeb

EZ WSI DicomWeb is a python library that provides the ability to effortlessly
extract an image patch from a pathology DICOM whole slide image (WSI) stored in
a cloud DICOM store. This functionality is useful for different pathology
workflows, especially those leveraging AI. EZ-WSI provides interfaces to enable embeddings to be easily generated from Digital Pathology patches and images.

This Github repository contains all source code, documentation, and unit tests
for EZ WSI DicomWeb. EZ WSI DicomWeb uses Bazel as a build and test tool.

## Requirements

- Python 3.10+

## Installing EZ WSI DicomWeb

1. Install necessary tools including [Bazel](https://bazel.build/install), and
[pip](https://pypi.org/project/pip/) (`sudo apt-get install pip`).

2. [Install the gcloud CLI](https://cloud.google.com/sdk/docs/install) and
login. Run the following commands from a shell. These will automatically accept
the [Google Cloud Platform TOS](https://cloud.google.com/terms).

    ```shell
    foo@bar$ gcloud auth application-default login
    foo@bar$ gcloud auth login
    foo@bar$ gcloud config set project <your_project>
    ```
3. Install EZ WSI DicomWeb. There are three recommended options for installing
EZ WSI DicomWeb. The first is preferred. The third option is best if you wish to
make modifications the library locally.

  - **Option 1:** Direct `pip` install.

      ```shell
      foo@bar$ pip install ez-wsi-dicomweb
      ```

  - **Option 2:** Direct `pip` install via Github.

      ```shell
      foo@bar$ pip3 install git+https://github.com/GoogleCloudPlatform/EZ-WSI-DICOMweb.git

  - **Option 3:** Clone the repository and manually install. This option is
  useful if you would like to make local modifications to the library and
  test them out.

      1. Run the following commands to make an installation directory for
      EZ WSI DicomWeb:

          ```shell
          foo@bar$ export EZ_WSI=$HOME/ez_wsi_install_dir
          foo@bar$ mkdir $EZ_WSI
          foo@bar$ cd $EZ_WSI
          ```

      2. Clone EZ WSI DicomWeb into the directory you just created:

          ```shell
          foo@bar$ git clone https://github.com/GoogleCloudPlatform/EZ-WSI-DICOMweb.git $EZ_WSI
          ```

      3. Install EZ WSI using `pip`.

          ```shell
          foo@bar$ pip install .
          ```

## Using EZ WSI DicomWeb

Check out the [Jupyter Notebook here](https://colab.sandbox.google.com/github/GoogleCloudPlatform/EZ-WSI-DICOMweb/blob/main/ez_wsi_demo.ipynb).

## Testing EZ WSI DicomWeb

You can execute unit tests using the following command:

```shell
foo@bar:ez_wsi_install_dir$ bazel test ez_wsi_dicomweb/ez_wsi_errors_test
```

Alternatively, you can run all of EZ WSI's unit tests by running

```shell
foo@bar:ez_wsi_install_dir$ bazel test //...
```

For more information on flags you can pass to Bazel, such as `--test_output=all`
see [here](https://docs.bazel.build/versions/2.0.0/command-line-reference.html).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

See [`LICENSE`](LICENSE) for details.