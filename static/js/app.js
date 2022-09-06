function toggle_display() {
  const imageSection = document.querySelector('.image-section');

  if (imageSection.childElementCount > 0) {
    const el = document.querySelector('.img-face');
    el.remove();
    return;
  }

  const image = document.createElement('img');
  image.className = 'img-face';
  image.src = '/face-detection';
  imageSection.append(image);
}
