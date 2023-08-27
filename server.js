const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');
const predictImageContents = require('./TensorFlow')

const app = express();

const storage = multer.memoryStorage()
const upload = multer({ storage: storage })

app.post('/image', upload.single('image'), async (req, res) => {
    const buffer = req.file.buffer;
    const decodedImage = tf.node.decodeImage(buffer);
    if (!/^image\/(jpe?g|png|gif)$/i.test(req.file.mimetype)) {
        console.log('Unsupported image type');
        res.json({ error: "Unsupported image type" })
        return;
    }
    const resizedImage = await sharp(req.file.buffer)
        .resize(224, 224)
        .toBuffer();

    const result = await predictImageContents(resizedImage);


    res.send(result);
});

app.set('view engine', 'ejs');
app.get('/', (req, res) => {
    res.render('index');
})

app.listen(3000, () => {
    console.log('Server listening on port 3000!');
});
