---
layout:     post
title:      "Using SDL2 in Rust"
date:       2023-09-17 00:00:00 +0200
categories: ["Rust", "Graphics", "GameDev"]
---

# Introduction

Simple DirectMedia Layer or just simple SDL is a cross-platform library used for accessing video, audio, input devices like keyboard, mouse or joysticks, in addition to also providing some networking abstractions. Despite its age of already 25 years, it is still used extensively for games and other multimedia software as an abstraction layer, either using it to directly draw graphics and play sounds or as a lower-level library on which games engines are built.

While written mainly in C, a lot of language bindings where created and one of them is the Rust binding [rust-sdl2](https://github.com/Rust-SDL2/rust-sdl2), which we will introduce here. We will show how to open a window, draw a small thing and use the events system from SDL2 to handle keyboard inputs.

# Linux setup

To get started with SDL2 in Rust, you first need the sdl2 library and headers installed on your system (the Rust crate has a bundled feature, where it compiles it from source, but we are not gonna talk about this here). Install the library using the appropriate package manager for your distribution.

Ubuntu:
```bash
sudo apt-get install libsdl2-dev
```

Fedora:
```bash
sudo dnf install SDL2-devel
```

Arch:
```bash
sudo pacman -S sdl2
```

# Windows setup

We assume you are using MSVC as your C++ compiler environment. if you are using MINGW, please see the crate documentation on how to continue.

1. Download the MSVC version of SDL2 from http://www.libsdl.org/ (usually named something like *SDL2-devel-2.x.x-VC.zip*).

2. Unzip *SDL2-devel-2.x.x-VC.zip*.

3. Copy all *.lib* files from `SDL2-2.x.x\lib\x64\` to `%userprofile%\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\x86_64-pc-windows-msvc\lib\`

4. Copy `SDL2-2.x.x\lib\x64\SDL2.dll` into your project directory or into any directory which is in your PATH. When you want to distribute the compiled application, make sure the ship `SDL2.dll` right next to it, or it may not run, if the user doesn't have the `SDL2.dll` lying around.

# Creating a Rust project

If you haven't installed it, yet, install Rust following the ["Install Rust"](https://www.rust-lang.org/tools/install) guide.

Create a new Rust project by typing 

```bash
cargo new sdl2-example
```

Then change into the newly created directory and type

```bash
cargo add sdl2 -F unsafe_textures
```

to add th SDL2 rust bindings.

We are going to use the `unsafe_textures` feature, even though we are not going to use any textures. Mainly because, if you do use textures, you will notice that without that option you are getting a lost of Rust [lifetime](https://doc.rust-lang.org/rust-by-example/scope/lifetime.html) issues. However, this comes with he downside, that you have to manage the texture objects yourself and make sure to call destroy, if you do not need them any longer. For more information about this feature see [here](https://doc.rust-lang.org/rust-by-example/scope/lifetime.html).

You can also add other features which correspond to the different optional SDL2 libraries like gfx, mixer or tff.

When this was successful, you are ready to start developing with SDL2!

# Opening a window

Let's open our `main.rs` file and start by creating a window inside the main function (note, we adapt the return value of the main function slightly):

```rust
use sdl2::{event::Event, keyboard::Keycode};

pub fn main() -> Result<(), String> {
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;

    let window = video_subsystem
        .window("rust-sdl2 example", 800, 600)
        .opengl()
        .build()
        .map_err(|e| e.to_string())?;

    let mut event_pump = sdl_context.event_pump()?;

    'main: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => {
                    break 'main
                },
                _ => {}
            }
        }
    }

    Ok(())
}
```

Now let's try to run the program
```bash
cargo run
```

If everything compiles and runs successfully, you will see an empty window. This is fine, we are not drawing anything, yet. But you should be able to close the program with the ESC key.

Here you can already see, that SDL2 is not a full game engine, you really have to do a lot of things yourself, like maintaining an event loop and mapping the events that SDL2 captures to something meaningful, like closing the program.

# Drawing a rectangle

We are going to use build-in drawing functionalities from SDL2. They are suitable for drawing simple primitives and we will use one of them to draw a rectangle.

```rust
use sdl2::{event::Event, keyboard::Keycode, pixels::Color, rect::Rect};

pub fn main() -> Result<(), String> {
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;

    let window = video_subsystem
        .window("rust-sdl2 example", 800, 600)
        .opengl()
        .build()
        .map_err(|e| e.to_string())?;

    let mut event_pump = sdl_context.event_pump()?;

    let mut canvas = window.into_canvas().build().map_err(|e| e.to_string())?;

    'main: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'main,
                _ => {}
            }
        }

        // Set the background
        canvas.set_draw_color(Color::RGB(255, 200, 0));
        canvas.clear();

        // Draw a red rectangle
        canvas.set_draw_color(Color::RGB(255, 0, 0));
        canvas.fill_rect(Rect::new(100, 100, 600, 400))?;

        // Show it on the screen
        canvas.present();
    }

    Ok(())
}
```

And that's it!

# Summary

As you can see, using SDL2 in Rust is as straightforward as in C and I hope this small introduction could serve as a starting point for your SDL2 and Rust adventures.

And here is a screenshot of our running program:

![rust2-sdl example window](/assets/img/sdl2_rust/image.png)
