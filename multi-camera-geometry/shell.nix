{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = [
    pkgs.python313
    pkgs.python313Packages.setuptools
    pkgs.python313Packages.matplotlib
    pkgs.python313Packages.tkinter
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${
      pkgs.lib.makeLibraryPath [
        pkgs.stdenv.cc.cc.lib
        pkgs.libxcb
        pkgs.libGL
        pkgs.glib
        pkgs.libx11
        pkgs.libXext
        pkgs.libSM
        pkgs.libICE
        pkgs.libxkbcommon
        pkgs.fontconfig
        pkgs.freetype
        pkgs.zlib
        pkgs.dbus
        pkgs.xcb-util-cursor
        pkgs.xcbutilimage
        pkgs.xcbutilkeysyms
        pkgs.xcbutilrenderutil
        pkgs.xcbutilwm
      ]
    }:$LD_LIBRARY_PATH"
  '';
}
