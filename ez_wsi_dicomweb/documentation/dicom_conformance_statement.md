# EZ-WSI DICOMweb DICOM Conformance Statement

EZ-WSI DICOMweb is a Python library that supports retrieval of DICOM imaging from Cloud based DICOM stores. The library also enables ML embeddings generation from DICOM imaging and other data sources. This  document outlines the DICOM conformance requirements for the EZ-WSI DICOMweb.


## DICOM Store requirements (Transport Protocol)

EZ-WSI DICOMweb requires that the DICOM store holding the imaging support the [DICOMweb](https://www.dicomstandard.org/using/dicomweb) REST API. Specifically, it requires support for JSON formatted [QIDO-RS](https://www.dicomstandard.org/using/dicomweb/query-qido-rs) and [WADO-RS](https://www.dicomstandard.org/using/dicomweb/retrieve-wado-rs-and-wado-uri) transactions over HTTPS. The URL for store transactions must end with a valid DICOMweb suffix.

Performing authenticated transactions using OAuth bearer tokens is highly recommended and supported. [Google DICOM](https://cloud.google.com/healthcare-api/docs/concepts/dicom) stores are compatible with the library.


## DICOM Store Data Organization Requirements

Data is expected to be organized within a DICOM store in accordance with the [DICOM Standard](https://www.dicomstandard.org/). The library expects the DICOM series containing [VL Whole Slide Microscopy Image IOD](https://dicom.nema.org/dicom/2013/output/chtml/part03/sect_A.32.html#sect_A.32.8) imaging will contain one representation of the image pyramid. DICOM imaging represented in either the  [VL Microscopic Image IOD](https://dicom.nema.org/dicom/2013/output/chtml/part03/sect_A.32.html#sect_A.32.2) or the [VL Slide-coordinates Microscopic Image IOD](https://dicom.nema.org/dicom/2013/output/chtml/part03/sect_A.32.html#sect_A.32.3) must not be represented in the same series as [VL Whole Slide Microscopy Image IOD](https://dicom.nema.org/dicom/2013/output/chtml/part03/sect_A.32.html#sect_A.32.8) imaging.  


### DICOM Data Requirements

The library supports image retrieval and embedding generation from DICOM imaging stored within a subset of DICOM IOD. This section outlines the library requirements of the supported DICOM imagery.


#### Supported DICOM Information Document Models (IOD)

The EZ-WSI DICOMweb supports embedding generation from DICOM images encoded in the following IODs. 


<table>
  <tr>
   <td><strong>Media Storage SOP Class UID /  SOP Class UID</strong>
   </td>
   <td><strong>Name</strong>
   </td>
  </tr>
  <tr>
   <td>1.2.840.10008.5.1.4.1.1.77.1.6
   </td>
   <td><a href="https://dicom.nema.org/dicom/2013/output/chtml/part03/sect_A.32.html#sect_A.32.8">VL Whole Slide Microscopy Image IOD</a>
   </td>
  </tr>
  <tr>
   <td>1.2.840.10008.5.1.4.1.1.77.1.2
   </td>
   <td><a href="https://dicom.nema.org/dicom/2013/output/chtml/part03/sect_A.32.html#sect_A.32.2">VL Microscopic Image IOD</a>
   </td>
  </tr>
  <tr>
   <td>1.2.840.10008.5.1.4.1.1.77.1.3
   </td>
   <td><a href="https://dicom.nema.org/dicom/2013/output/chtml/part03/sect_A.32.html#sect_A.32.3">VL Slide-coordinates Microscopic Image IOD</a>
   </td>
  </tr>
</table>



##### ***VL Whole Slide Microscopy Image IOD*  (1.2.840.10008.5.1.4.1.1.77.1.6)**

[VL Whole Slide Microscopy Imaging IOD](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6236926/) is the DICOM IOD that represents gigapixel multi-frame whole slide imaging as captured by commercial slide scanners. Digital pathology images encoded within this IOD are typically represented in DICOM as a pyramid of DICOM instances that encode the imaging at multiple magnifications. In addition to this, instances may be encoded to represent the slide label, a thumbnail image, and area of the image acquired by the slide scanner. At a high level the embedding service expects that processed DICOM is conformant with the [DICOM Standard](https://www.dicomstandard.org/), e.g., [Type-1 tags](https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_7.4.html) are defined and all tags have values that conform to their respective requirements (e.g., [Value Representation (VR)](https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_6.2.html) and [Value Multiplicity (VM)](https://dicom.nema.org/dicom/2013/output/chtml/part05/sect_6.4.html) codes).  Additionally to this the following attributes are required.

**[Supported Transfer Syntax](https://dicom.nema.org/medical/dicom/current/output/chtml/part18/sect_8.7.3.html)**


<table>
  <tr>
   <td><strong>Recommended Transfer Syntax UID</strong>
   </td>
   <td><strong>Name</strong>
   </td>
  </tr>
  <tr>
   <td>1.2.840.10008.1.2.4.50 <strong>(Preferred)</strong>
   </td>
   <td>JPEG Baseline (Process 1):
<p>
Default Transfer Syntax for Lossy JPEG 8-bit Image Compression
   </td>
  </tr>
  <tr>
   <td>1.2.840.10008.1.2.4.90
   </td>
   <td>JPEG 2000 Image Compression (Lossless Only)
   </td>
  </tr>
  <tr>
   <td>1.2.840.10008.1.2.4.91
   </td>
   <td>JPEG 2000 Image Compression
   </td>
  </tr>
</table>


Transfer syntax’s other than those listed above are transcoded to 1.2.840.10008.1.2.1,` `uncompressed little endian, using  the DICOM store transcoding and will be limited by the capabilities of the DICOM store hosting the imaging. Google DICOM Store’s [transcoding capabilities are listed here](https://cloud.google.com/healthcare-api/docs/dicom#supported_transfer_syntaxes_for_transcoding).

**Encapsulated Transfer Syntaxes**

Multi-frame DICOM instances that encode encapsulated a transfer syntax are required to represent each frame's imaging within a single fragment. DICOM instances that encode an encapsulated transfer syntax and do not contain frames are required to represent encapsulated pixel data within a single fragment.

**ICC Color Profile**

If an ICC Profile is defined. It is expected that the same ICC profile will apply to all levels of the image pyramid.

**Image Pixel Module:**

[Samples Per Pixel (0028, 0002)](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.3.html#table_C.7-11a)

All imaging is required to be encoded 1 or 3 samples per pixel.  

For monochrome or gray scale imaging:  SamplesPerPixel=1

For color imaging:  SamplesPerPixel=3

[Bits Allocated (0028, 0100)](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.3.html#table_C.7-11a)

All imaging is required to be encoded with 8 bits per sample, channel, **BitsAllocated = 8** 

[High Bit (0028, 0102)](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.3.html#table_C.7-11a)

All imaging is required to be encoded in little endian with **HighBit = 7**

**Multi-frame Dimension Module**

[Dimensional Organization Type (0020, 9331)](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.17.html#table_C.7.6.17-1)

All multi-frame instances are required to be defined with  **DimensionOrganizationType = TILED_FULL. **

**Multiple Focal Planes Are Not Supported**

Images encoded with multiple focal planes are not supported.

**[Optical Path Sequence ](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.12.5.html#table_C.8.12.5-1)**

Images are expected to be acquired using a single optical path and have one optical path sequence.

**Required Tags**


<table>
  <tr>
   <td>(0002, 0010) TransferSyntraxUID
<p>
(0008, 0008) ImageType
<p>
(0008, 0016) SOPClassUID
<p>
(0008, 0018) SOPInstanceUID
<p>
(0008, 0060) Modality
<p>
(0020, 000E) SeriesInstanceUID
<p>
(0020, 000D) StudyInstanceUID
<p>
(0020, 9161) ConcatenationUID   C
<p>
(0020, 9228) ConcatenationFrameOffsetNumber  C
<p>
(0020, 9331) DimensionalOrganizationType
<p>
(0028, 0002) SamplesPerPixel
   </td>
   <td>(0028, 0010) Rows
<p>
(0028, 0011) Columns
<p>
(0028, 0100) BitsAllocated
<p>
(0028, 0102) HighBit
<p>
(0028, 2000) ICC Profile  P
<p>
(0028, 9110) PixelSpacing *
<p>
(0048, 0001) ImagedVolumeWidth
<p>
(0048, 0002) ImagedVolumeHeight
<p>
(0048, 0006) TotalPixelMatrixColumns
<p>
(0048, 0007) TotalPixelMatrixRows
<p>
(7FE0, 0010) PixelData
   </td>
  </tr>
</table>


**C **Tags are conditionally required if a DICOM instance is defined as part of instance concatenation (not common).

**P** ICC_Profile metadata is a non root level DICOM tag. Tag is expected to be defined within the OpticalPathSequence.

(0048,0105) OpticalPathSequence -> (0028, 2000)  ICC_Profile

***** PixelSpacing is a non-root level DICOM tag. The tag is within the following nested sequence.

(5200, 9229) Shared functional group sequence -> (0028, 9110) Pixel Measures Sequence -> (0028, 9110) PixelSpacing 


##### ***VL Microscopic Image IOD*  (**1.2.840.10008.5.1.4.1.1.77.1.2**) and *VL Slide-coordinates Microscopic Image IOD*  (**1.2.840.10008.5.1.4.1.1.77.1.3**)**

[VL Microscopic Image IOD](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.32.2.3.html#table_A.32.1-2) and VL [Slide-coordinates Microscopic Image IOD](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.32.3.3.html#table_A.32.1-3) are the DICOM IOD that represents non-tiled microscope imaging such as what is captured by a microscope attached digital camera. Images encoded in these IODs are represented by a single instance. At a high level the embedding service expects that processed DICOM is conformant with the [DICOM Standard](https://www.dicomstandard.org/), e.g., [Type-1 tags](https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_7.4.html) are defined and all tags have values that conform to their respective requirements (e.g., [Value Representation (VR)](https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_6.2.html) and [Value Multiplicity (VM)](https://dicom.nema.org/dicom/2013/output/chtml/part05/sect_6.4.html) codes).  Additionally to this the following attributes are required.

**[Supported Transfer Syntax](https://dicom.nema.org/medical/dicom/current/output/chtml/part18/sect_8.7.3.html)**


<table>
  <tr>
   <td><strong>Recommended Transfer Syntax UID</strong>
   </td>
   <td><strong>Name</strong>
   </td>
  </tr>
  <tr>
   <td>1.2.840.10008.1.2.4.50 <strong>(Preferred)</strong>
   </td>
   <td>JPEG Baseline (Process 1):
<p>
Default Transfer Syntax for Lossy JPEG 8-bit Image Compression
   </td>
  </tr>
  <tr>
   <td>1.2.840.10008.1.2.4.90
   </td>
   <td>JPEG 2000 Image Compression (Lossless Only)
   </td>
  </tr>
  <tr>
   <td>1.2.840.10008.1.2.4.91
   </td>
   <td>JPEG 2000 Image Compression
   </td>
  </tr>
</table>


Transfer syntax’s other than those listed above are transcoded to 1.2.840.10008.1.2.1,` `uncompressed little endian, using  the DICOM store transcoding and will be limited by the capabilities of the DICOM store hosting the imaging. Google DICOM Store [transcoding capabilities](https://cloud.google.com/healthcare-api/docs/dicom#supported_transfer_syntaxes_for_transcoding).

**Encapsulated Transfer Syntaxes**

Multi-frame DICOM instances that encode encapsulated a transfer syntax are required to represent each frame's imaging within a single fragment. DICOM instances that encode an encapsulated transfer syntax and do not contain frames are required to represent encapsulated pixel data within a single fragment.

**Image Pixel Module:**

[Samples Per Pixel (0028, 0002)](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.3.html#table_C.7-11a)

All imaging is required to be encoded 1 or 3 samples per pixel.  

For monochrome or gray scale imaging:  SamplesPerPixel=1

For color imaging:  SamplesPerPixel=3

[Bits Allocated (0028, 0100)](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.3.html#table_C.7-11a)

All imaging is required to be encoded with 8 bits per sample, channel, **BitsAllocated = 8** 

[High Bit (0028, 0102)](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.3.html#table_C.7-11a)

All imaging is required to be encoded in little endian with **HighBit = 7**

**[Optical Path Sequence ](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.12.5.html#table_C.8.12.5-1)**

Images are expected to be acquired using a single optical path and have one optical path sequence.

**Required Tags**


<table>
  <tr>
   <td>(0002, 0010) TransferSyntraxUID
<p>
(0008, 0008) ImageType
<p>
(0008, 0016) SOPClassUID
<p>
(0008, 0018) SOPInstanceUID
<p>
(0020, 000E) SeriesInstanceUID
<p>
(0020, 000D) StudyInstanceUID
<p>
(0028, 0002) SamplesPerPixel
   </td>
   <td>(0028, 0010) Rows
<p>
(0028, 0011) Columns
<p>
(0028, 0100) BitsAllocated
<p>
(0028, 0102) HighBit
<p>
(0028, 2000) ICC Profile  P
<p>
(0028, 9110) PixelSpacing
<p>
(7FE0, 0010) PixelData
   </td>
  </tr>
</table>


**P** ICC_Profile metadata is a non root level DICOM tag. Tag is expected to be defined within the OpticalPathSequence.

(0048,0105) OpticalPathSequence -> (0028, 2000)  ICC_Profile
