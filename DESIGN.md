---
name: simple-rag
description: The interface of a local RAG application: Streamlit's theme, the Russian wording, four relevance ratings.
colors:
  rating-excellent: "green"
  rating-good: "blue"
  rating-medium: "orange"
  rating-low: "red"
typography: {}
rounded: {}
spacing: {}
---

# Design system: simple-rag

## Overview

**Creative North Star: "The instrument panel"**

simple-rag draws almost nothing of its own. The interface is a Streamlit
application on the operator's machine, and Streamlit's default theme supplies
the page, the type, the buttons and the panels. There is no stylesheet, no
theme file, no colour constant and no font name anywhere in the repository.

What the project does own is a smaller thing: the wording, the glyph that
opens every label, the four relevance ratings and their thresholds, and the
shape of the panel that shows the sources behind an answer. Those decisions
are in `streamlit_rag.py`, and they are the whole visual language of the
product.

The interface is in Russian. The tab titles, the buttons, the counters, the
log lines and the error messages are Russian, and so are the comments in the
source and the commit messages. The documents in this repository are in
English, so the two have different audiences.

The rejections are as deliberate as the decisions. There is no custom CSS and
no `.streamlit/config.toml`, so there is no dark-mode toggle and no palette.
There is no logo and no illustration. There is no second font. Adding any of
them means the interface stops being Streamlit's, and the operator's next
upgrade of Streamlit changes the look regardless.

**Key characteristics:**

- the Streamlit theme, untouched
- Russian wording, and a glyph in front of every label
- four relevance ratings, four words, four colours, three decimal places
- the answer is never shown without its sources
- the run log sits under the grid, not in a file the operator has to open

## Colors

The project names four colours, and the code names them as the Streamlit
theme's own names, in the chat panel, when it rates a source.

- **Excellent** (`green`): a score of 0.8 and above, the word "отличная".
- **Good** (`blue`): a score from 0.6, the word "хорошая".
- **Medium** (`orange`): a score from 0.4, the word "средняя".
- **Low** (`red`): a score below 0.4, the word "низкая".

Every other colour in the interface belongs to Streamlit: the ground, the
text, the primary button, the info, warning and error boxes. The project
never names those, so there are no hex values to record here.

### Named rules

**The Rating Is Doubled Rule.** A source is rated with a word, a number to
three decimal places, and a colour. The word is what a reader reads on a
screen that renders no colour at all, and the number is what a reader needs
when two fragments from different files score close to each other.

**The Threshold Rule.** The four cut-points are 0.8, 0.6 and 0.4, and they
are hard-coded in one place in `streamlit_rag.py`. Nothing measures them and
nothing configures them. A fifth band would be a different system rather than a
variant of the four.

## Typography

The project sets no typeface, and names none. Streamlit's theme decides the
body text, the headings and the code, and the application never overrides it.

There is one typographic decision the project does make: the log console is
rendered with `st.code(console_text, language="text")`, so it is the one
block on the screen that is monospaced. Everything else in the interface is
proportional.

### Named rules

**The Console Is The One Monospace Rule.** The run log is the only block that
looks like a terminal. It is there to be read as a sequence of events, with
timestamps in a fixed column.

## Layout

The page is set to `layout="wide"`, and it holds a sidebar, two tabs and
nothing else.

The sidebar carries three counters and nothing else: documents found,
documents indexed, and chat messages sent. Each counter shows a delta
against the previous run of the page, and the delta appears only after the
second run, because there is nothing to compare the first one with.

The first tab is the index. Its order is fixed: the file grid, the line that
reports how many files are selected, a divider, the two action buttons side
by side, another divider, the log console, and the button that clears the
log. The console is always visible, underneath the grid, whether or not an
indexing run is in progress.

The second tab is the chat. When nothing is indexed yet, the tab shows a
warning and stops. When there is an index, it shows the conversation, and
under every answer an expander holding the sources behind it.

### Named rules

**The Console Sits Under The Grid Rule.** The run log is part of the index
screen, not a separate view and not a file. The operator reads the grid and
the log in the same look, at the same time.

**The Sources Sit Under The Answer Rule.** The expander that holds the
sources belongs to the message, directly under it. A reader checks an answer
against the fragment it came from, and the two are never more than one scroll
apart.

## Elevation & Depth

Depth belongs to Streamlit. There is no custom CSS, no shadow, no overlay
and no z-index rule in the repository.

The one depth cue the project asks for is the expander, which is Streamlit's
own: the sources are folded away by default and open on request, so the
conversation stays readable when the fragments are long.

## Shapes

Radii belong to Streamlit. The project names no radius and sets no border.
The rounded corners on the buttons, the metrics and the expander are the
theme's.

## Components

### The file grid

- **Shape:** an AG Grid table with checkbox selection, four columns, and the
  grid's own header row.
- **Columns:** four, all in Russian: the folder, the file name with its
  type glyph, the type, and whether the file is indexed. The file name
  carries a different glyph per type: a book for PDF, another book for DOC
  and DOCX.
- **State:** the "indexed" column reads from the Chroma collection, through
  the set of source paths it holds.
- **Selection:** multiple, with checkboxes. The button below it is disabled
  while nothing is selected.

### The action buttons

- **Primary:** "Индексировать выбранные", in the left column. Disabled when
  the selection is empty and while a run is in progress.
- **Secondary:** "Очистить индекс", in the right column. It deletes the
  collection and reloads the page.
- **Glyphs:** a down arrow into a tray for indexing, a wastebasket for
  clearing.

### The log console

- **Style:** a code block under the grid, with the timestamp in square
  brackets at the left of every line.
- **Content:** the last 15 lines of the last 20 lines the app keeps. The
  console holds 20, and shows 15.
- **Glyphs:** every line carries one, and the glyph says what happened: a
  magnifier for reading a file, a floppy for the store, a broom for clearing,
  a cross for a failure.

### The source panel

- **Shape:** an expander under the answer, titled with the number of
  sources it holds.
- **Contents:** for each source, the file name in bold, the rating word in
  its colour with the score to three decimals, the fragment in italics, and
  a divider between sources.

### The counters

- **Shape:** three metrics in the sidebar, in the order documents found,
  documents indexed, chat messages.
- **Delta:** shown against the previous page render, and hidden on the first
  one.

## Do's and Don'ts

- **Do** keep the four rating bands and their three thresholds in the one
  place they live in now, and change the word and the colour together.
- **Do** open every label with the same glyph for the same action, so the
  icon is readable on its own.
- **Do** show the source panel with every answer, including the answers
  that failed.
- **Do** leave the Streamlit theme alone unless a decision has been written
  down first.
- **Don't** add a stylesheet, a `.streamlit/config.toml` or a custom
  theme. The interface is Streamlit's, and a second layer on top of it has
  to be maintained across Streamlit upgrades.
- **Don't** add a fifth rating band or move a threshold without changing
  the word that goes with it.
- **Don't** rename a column of the grid without changing the key the
  selection reads, because the selection stops being readable at once.
- **Don't** print an answer without the file, the score and the fragment
  behind it.
