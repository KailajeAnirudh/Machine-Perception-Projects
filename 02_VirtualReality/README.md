# Virtual Reality through PnP and P3P Algorithm Implementation

To run this program:

```
cd code
python main.py
```


We also provided some helper flags. Please check `main.py` for details. You can generate your visualizations with either PnP or P3P algorithm, both of which have been implemented. 

## Results
![Virtual Reality](code\Result.gif)
## Debugging

It's recommended to run the program with `--debug` to understand the program. 

Also, note that the main program has several other args you can set, please have a look at line 40 in the `main.py` for more details: You can pass `--solver PnP` or `--solver P3P` to the program to toggle between the solving methods.

Note, we also provide the `.vscode` launch configuration for you to easily debug in VSCode.

## Customization

You need to assign different values to  `click_point` in `main.py` to render the drill at different places. 



