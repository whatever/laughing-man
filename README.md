# :hocho:

## Start capture images and filtering out laughing person's face

```bash
pip3 install "xD @ git+ssh://github.com/whatever/laughing_person/"
xD
```

## Example workflow

### Install package from github

```bash
pip install "xD @ git+ssh://github.com/whatever/laughing_person/"
```

### Capture images of self to use a positive examples

```bash
xD capture --device 0
```

### Download more images to use as negative examples

```bash
wget ...
```

### Augment images (flip, crop, shift RGB, etc.)

```bash
xD split
```

### Train on generated data set

```bash
xD train --epochs 10 --checkpoint result.pt
```

### Benchmark on validation set

```bash
xD benchmark --checkpoint result.pt --validate-dir augmented-data/validate/ 
```

### Live capture video and filter out positive examples

```bash
xD --checkpoint result.pt --device 1
```
