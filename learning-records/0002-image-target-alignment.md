# Spatial transformations must preserve image–target alignment

The user correctly predicted that horizontally flipping an image without transforming its target leaves the person's bounding box on the wrong side. This demonstrates understanding of why spatial transformations must update images and annotations together. Next, trace the dataset return values into batch collation.
