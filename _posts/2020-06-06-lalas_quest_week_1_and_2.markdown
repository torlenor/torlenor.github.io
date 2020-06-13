---
layout: post
title:  "Lala's Quest Week 1 & 2"
date:   2020-06-06 18:00:00 +0200
categories: "Lala's quest"
---
A roguelike exploration game featuring a little cat, monsters and mutations.

Due to Corona I had more time than usual for my private projects. As I had no motivation to keep on coding on my other projects, I started a new one: A game! I stumbled across the topic of roguelikes again, which I did not play in a long time (except games like FTL). As my other tries to do something game-like was usually always messing around with graphics, 3d models and materials in Unreal Engine 4, I thought this time I will give pure ASCII a try as this would for sure be a lot less complex. Well, was I wrong! Over the last two weeks I discovered the topics of FoV calculations, pathfinding, procedural map generation, structuring of code for games, e.g., ECS, and many more things.

I decided to write it from scratch in Go with go-sdl2 for graphics output. I know there would be approaches were I would not have to do everything by hand, but as I am mostly interested in the technical aspects, this is the way to go for me.

The sales pitch for the game would be: A little cat goes for a mouse hunt and ends up in a big adventure filled with monsters and genetic manipulation.

What I am currently working on is the inventory system and the mutation system (which will be the main feature). Certain elements of the game will only be available with certain mutations, but you cannot consume all of them, you have to decide. For example one of the mutations will give you an inventory (or how would you expect a little cat to be able to carry things, except through genetic manipulation?).

I am currently struggling with the core turn based game system and the entity management and to keep everything organized, so that it is easy to add additional items and mutagen mechanics. Also the performance on potato systems is not great, so I have to look into that at some point. Still a lot to learn.

I will open-source it when the code is a little less messy.

Oh and here is a WIP gif:
![](https://i.imgur.com/6Kf9Yci.gif)