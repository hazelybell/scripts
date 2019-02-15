#!/bin/bash

DEST=/opt
APPDEST=/usr/local/share/applications
ICONDEST=/usr/local/share/pixmaps
TARURL='https://download.mozilla.org/?product=firefox-devedition-latest-ssl&os=linux64&lang=en-US'
ICON="firefox/browser/chrome/icons/default/default128.png"

cd $DEST || exit 1
echo "Uninstalling old firefox..."
rm -rvf firefox
echo "Downloading and unpacking..."
curl -L $TARURL | tar -xvj --overwrite -f - || exit 2

mkdir -p $APPDEST || exit 3
cd $APPDEST

cat >|firefox_dev.desktop <<EOF || exit 4
[Desktop Entry]
Name=Firefox Developer
GenericName=Firefox Developer Edition
Exec=/opt/firefox/firefox
Terminal=true
Icon=/usr/local/firefox_dev/browser/icons/mozicon128.png
Type=Application
Categories=Application;Network;X-Developer;
Comment=Firefox Developer Edition Web Browser.
Icon=firefox_dev
EOF

mkdir -p $ICONDEST || exit 5
cd $ICONDEST
cp $DEST/$ICON firefox_dev.png || exit 6
