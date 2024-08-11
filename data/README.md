The file 'dataset.csv' contains the final dataset.  
The most relevant colums are:  
* `unsafe_prompt`: original English prompts identified as unsafe, violating categories like "Violence" or "Disinformation"
* `unsafe_completion`: English model completions generated in response to unsafe prompts in `unsafe_prompt`, flagged as either fully or partially unsafe.
* `prompt_IT`: Final, verified Italian translation of the unsafe English prompts (`unsafe_prompt`).
* `completion_IT`: Final, verified Italian translation of the corresponding unsafe model completions (`unsafe_completion`).

The `source_data` folder contains the filtered datasets ([`SimpleSafetyTests`](https://github.com/bertiev/SimpleSafetyTests) and [`StrongReject`](https://github.com/alexandrasouly/strongreject)) from which prompts were taken.