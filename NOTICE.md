# Third-party attributions

Constellation incorporates ideas and design elements from the following
third-party projects. The list grows as the project absorbs more
external work.

## Visual design

The Constellation dashboard and genome browser color palette is inspired
by the **Aura theme** by Dalton Menezes
(<https://github.com/daltonmenezes/aura-theme>), MIT-licensed. Accent
selection draws additional inspiration from the **Aura port for Raycast**
by Shubh Porwal (<https://ray.so/themes/shubh_porwal/aura>). Color values
are adapted for Constellation's elevation needs, not copied verbatim.

The relevant files are
[`constellation/viz/frontend/index.dashboard.html`](constellation/viz/frontend/index.dashboard.html)
and
[`constellation/viz/frontend/index.genome.html`](constellation/viz/frontend/index.genome.html)
(`:root` CSS variable blocks).

## Mass-spectrometry models

Constellation runs the TorchScript models of **JChronologer**
(<https://github.com/searlelab/jchronologer>, Apache-2.0, Searle lab),
the library predictor EncyclopeDIA ships, by reading them from the
user's installed EncyclopeDIA; it does not redistribute the models. The
tokenizer in
[`constellation/massspec/tokenize/jchronologer.py`](constellation/massspec/tokenize/jchronologer.py)
ports JChronologer's preprocessing rules. Four of its model
preprocessing files (`*.preprocessing.json`) are copied unmodified under
[`tests/data/jchronologer/`](tests/data/jchronologer/) as test fixtures,
under the Apache License 2.0
(<https://www.apache.org/licenses/LICENSE-2.0>); copyright remains with
their authors.
