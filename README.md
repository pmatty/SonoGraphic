# SonoGraphic

SonoGraphic creates bar-aligned sonograms from a fixed-media audio file. It can render an overview of the whole piece, export an individual image for each bar, and (optionally) align those images with the bar positions of a `Tape` staff in a score PDF.

The application is a desktop GUI adaptation of `SonoGraphic_workbook.ipynb`. It is written in Python and uses PySide6 for the interface.

## Features

- Read audio from WAV, AIFF and FLAC files using `soundfile`.
- Select and average one or more audio channels.
- Derive bar boundaries from a MIDI tempo map, a CSV tempo map, or a constant tempo and metre.
- Apply an audio offset to align the recording with the tempo map.
- Generate a four-part overview sonogram with barlines, bar numbers and optional rehearsal marks.
- Export one sonogram image per bar, together with a CSV manifest of bar timings and image widths.
- Optionally read the `Tape` staff from a score PDF, detect its systems and barlines, and create score-aligned sonogram strips.
- Optionally stamp the strips onto the original score PDF.
- Save a method record describing the analysis settings.

## Requirements

- Python 3.10 or later
- The Python packages listed below

The application has been developed as a Python desktop application. On macOS, use a Python installation that matches your machine's architecture. On an Apple Silicon Mac, an ARM-native Python installation is generally preferable to running an x86-64 interpreter through Rosetta.

## Installation

It is recommended that you create a virtual environment for the project.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib pillow soundfile mido pdfplumber pymupdf PySide6
```

### Windows

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib pillow soundfile mido pdfplumber pymupdf PySide6
```

If you do not need score-PDF alignment, `pdfplumber` and `pymupdf` are only needed for the optional PDF stage, although the current installation command includes them for completeness.

## Running the application

Save `sonographic_gui.py` in your project directory, activate the virtual environment, and run:

```bash
python sonographic_gui.py
```

The window has two main areas:

- **Inputs and settings** on the left: choose files and configure the tempo map, score alignment, sonogram analysis and image output.
- **Analysis outputs** on the right: completed stages appear as the analysis runs. The process log and progress bar are shown beneath the output area.

The analysis runs in a background thread, so the interface should remain responsive while the sonograms are being generated.

## Preparing the inputs

### Audio file

Choose a WAV, AIFF or FLAC file containing the fixed-media part.

The channel field accepts comma-separated, one-based channel numbers. For example:

- `1` analyses the first channel.
- `1,2` averages the first and second channels.
- `1,2,3,4` averages the first four channels.

The selected channels are averaged to produce the mono signal used for analysis. Make sure that the channel numbers exist in the audio file.

### Tempo map

Choose one of three sources.

#### MIDI

Select **MIDI** and provide a MIDI file containing the tempo and time-signature information exported from your notation software.

The MIDI tempo map is used to calculate the start time of each bar. The application assumes a default tempo of 120 quarter-note beats per minute and a default time signature of 4/4 until changes are encountered in the file.

#### CSV

Select **CSV** and provide a comma-separated file containing bar numbers and their start times in seconds. An optional `end` row specifies the final boundary. If no `end` row is supplied, the audio duration (after applying the offset) is used.

Example:

```csv
bar,start_s
1,0
2,4.25
3,8.50
4,13.00
end,17.25
```

The bar numbers identify the boundaries in the file; the start times are measured in seconds from the beginning of the aligned audio timeline. Rows beginning with `#` are ignored. A header row is also ignored.

#### Constant

Select **Constant** and enter:

- tempo in quarter-note beats per minute;
- metre, such as `4/4`, `3/4` or `6/8`;
- number of bars.

Bar durations are calculated from the tempo and metre. This option is intended for music with a constant tempo and time signature.

### Score PDF (optional)

A score PDF is optional. If supplied, SonoGraphic attempts to locate a staff labelled `Tape`, detect its horizontal staff line and vertical barlines, and use those positions to construct a sonogram strip for each system.

For this stage to work, the PDF needs to contain extractable vector line and text information. A scanned or flattened PDF may not provide the geometry required for automatic detection. The staff label is currently expected to be exactly `Tape` and to appear towards the left of the system.

The PDF stage is therefore dependent on the engraving and structure of the score. Check the generated system images and the stamped PDF rather than assuming that automatic detection has been successful.

## Settings

The interface groups the controls into the following sections.

### Tempo map

- **Source**: MIDI, CSV or Constant.
- **Constant tempo (qpm)**: quarter-note beats per minute, used in Constant mode.
- **Metre**: time signature, used in Constant mode.
- **Number of bars**: number of bars to generate, used in Constant mode.

### Score and timing

- **First bar number**: the number assigned to the first bar in the tempo map.
- **Audio offset (s)**: shifts the audio relative to the tempo map. A positive value removes that amount from the beginning of the audio; a negative value pads silence at the beginning.
- **Expected duration**: optional duration in seconds used for a timing check.
- **Score check bar**: bar number used to choose a page for the PDF preview.
- **Rehearsal marks**: optional marks in the form `A:1, B:25, C:49`, where each letter is a mark and each number is its bar number.

### Sonogram settings

- **Channels to average**: comma-separated, one-based audio channel numbers.
- **FFT window**: number of samples in the analysis window.
- **Hop size**: number of samples between successive analysis frames.
- **Minimum frequency (Hz)** and **Maximum frequency (Hz)**: frequency limits of the logarithmic frequency axis.
- **Dynamic range (dB)**: display range below the peak level of the complete audio file.
- **Spectral tilt (dB/oct)**: optional frequency-dependent adjustment.
- **Gamma**: controls the contrast response of the sonogram.

### Image and engraving

- **DPI**: resolution used for exported bar and system images.
- **Image height (mm)**: height of each bar image and score-aligned strip.
- **Bar width (mm/s)**: horizontal scale for individual bar images, expressed as millimetres per second.
- **Style**: `ink` or `transparent`.
- **Matplotlib colour map**: optional colour map for rendering. Leave blank for the default monochrome or transparent rendering.
- **System first-bar inset (pt)**: horizontal inset applied at the beginning of a score system.
- **System barline width (px)**: optional barline drawn into the score-aligned strip.
- **Overview bar-number interval**: interval at which bar numbers are shown in the whole-piece overview.

## Outputs

All outputs are written to the selected output directory.

| File | Description |
| --- | --- |
| `bar_durations.png` | Plot of the duration of each bar, useful for checking the tempo map. |
| `overview.png` | Four-part overview sonogram of the complete piece, with bar boundaries and optional rehearsal marks. |
| `bar_XXX.png` | Individual sonogram image for each bar. The numbering begins at the configured first bar number. |
| `manifest.csv` | Bar number, start and end times, duration, image filename and image width. |
| `bar_images_preview.png` | Contact sheet showing a preview of the first four individual bar images. |
| `system_XX_bars_XXX-XXX.png` | Sonogram strip for each detected score system, created when a score PDF is supplied. |
| `systems.csv` | Information about the detected score systems and their bar ranges. |
| `system_images_preview.png` | Contact sheet of the generated score-system strips. |
| `score_with_strips.pdf` | Copy of the supplied score PDF with the sonogram strips stamped onto it. |
| `method.txt` | Text record of the principal analysis settings and tempo-map source. |

The system images, `systems.csv`, system preview and stamped PDF are only produced when a score PDF is supplied and the PDF stage succeeds.

## Suggested workflow

1. Export the fixed-media audio to WAV, AIFF or FLAC.
2. Export a MIDI file containing the score's tempo and time-signature information, or prepare a CSV tempo map.
3. Open SonoGraphic and select the audio and tempo-map source.
4. Set the first bar number and adjust the audio offset if the recording does not begin at the first bar boundary.
5. Check the bar-duration plot and the whole-piece overview.
6. If using a score PDF, inspect the system preview and stamped PDF carefully.
7. Review the individual bar images and `manifest.csv` in the output directory.

A warning in the process log that the tempo-map duration and audio duration differ is a prompt to check the MIDI/CSV export and audio offset. It is not automatically corrected by the application.

## Notes and limitations

- The sonogram is calculated from the mean of the selected channels. It is not a multichannel spatial analysis.
- The frequency axis is logarithmic. Magnitudes are displayed relative to the peak of the complete analysed file.
- Dark areas represent higher levels in the default monochrome `ink` rendering.
- Automatic score alignment depends on the PDF's underlying vector geometry and text extraction. It may need adjustment for different notation software, page layouts or staff labels.
- The MIDI tempo-map reader is intended for conventional exported tempo and time-signature maps. Inspect the bar-duration plot and compare the final tempo-map time with the audio duration, particularly where tempo or metre changes occur.
- High DPI values and large image heights can produce large images and increase memory use. For an initial test, use a lower DPI and increase it once the alignment and appearance are satisfactory.
- This is a research and engraving aid. Check the generated images, bar boundaries and PDF alignment against the score and audio before using them in a final publication or performance material.

## Project files

```text
SonoGraphic/
├── SonoGraphic_gui.py
├── SonoGraphic_workbook.ipynb
├── README.md
```



## Licence
