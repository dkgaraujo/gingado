# Welcome to the `gingado` contributing guideline <!-- omit in toc -->

Thank you for you interest in contributing to `gingado`! Your contribution is very welcome.

## Report issues, share suggestions and propose new functionalities

If you found a bug, want to share specific suggestions and even propose new functionalities, please [create a new issue](https://github.com/dkgaraujo/gingado/issues).

## Propose changes or additions to the library

If you want to contribute changes to the codebase (including also documentation and tests), you need to install [`nbdev`](https://nbdev.fast.ai/tutorials/tutorial.html). `gingado` is built using this library because of its expressiveness and ease of building complete solutions.

Note that in `nbdev`, all changes to the codebase go through notebooks, ie the .py files are not directly edited.

If a new code functionality is added, consider also document it and include testing code as mentioned below.

### Documentation

Please document new functions or classes using the `docments` feature, which is already present in `nbdev`.

Using `docments` is as easy (a) as declaring the expected types of each argument, if any; (b) writing a short description of the argument in the header; and (c) declaring the expected output of the function.

### Tests

Testing usually is fairly simple: use the new code in the notebook with one or more examples. These examples will be run during new commits, ensuring the new code is tested.
