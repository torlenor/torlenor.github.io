---
layout: post
title:  "Lala's Quest | Week 4"
date:   2020-06-21 08:25:00 +0200
categories: ["Lili's quest", "Lala's quest"]
---
During this week I worked on refactoring the input handling, to make it easier adding new keyboard shortcuts. This also has the advantage, that I can disentangle the SDL events from the actual input handling in the program. Later on I have to think about a system on how to have different input key bindings/behaviors depending on the state of the game (main menu, options menu, inventory modal open, for example), but this is something I have to think of when I actually have additional game states.

In addition, I was not happy with the rendering. I wanted to have the ability to render directly onto a grid and therefore I added a console. For now there is only MatrixConsole which can work with square or rectangle fonts (e.g., 12x12 or 6x12) that form a grid. In the same manner as with libtcod you can then put chars onto those grids and customize their foreground and background color. The game map is using this now, which makes rendering much simpler. Due to that refactoring/rewrite I also moved the rendering of the entities into the game map, which makes more sense to me than in the actual game logic.

A simple main menu is now implemented. It can only start the game or quit the application and has placeholders for Options and Load Game. It uses the same MatrixConsole as the map, just with a different font texture. Ideally I can use a Text Console later on, but for now it is good enough to get the logic of the menus set up.

Oh, and here is a new GIF:

![](https://i.imgur.com/P0M4eYA.gif)