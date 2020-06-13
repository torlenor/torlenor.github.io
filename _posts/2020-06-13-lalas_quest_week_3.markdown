---
layout: post
title:  "Lala's Quest Week 3"
date:   2020-06-13 19:00:00 +0200
categories: "Lala's quest"
---
After the first two weeks and now well into the third week I am starting to get a feeling where I have to go with the code structure to make it work. This week I got the "use" system for the mutagens (the items which cause mutations) and for items to work. Not sure if I should combine those at some point and make them all just carry effects, but we will see where it goes from here. Also due to a lot of refactoring these things are now acting as "systems" on all entities, which should be the first step to not handle the player in a special way. Position changes are still handled in a messy way, though.

Small things: I am also working on getting the UI separated from the main "game" package, to make it easier to update the UI later on. Also simple room changes are now working via portals/stairs/"+"-symbols on the map.

TODOs:
- More refactoring to make the code easier to extend.
- Move position change logic into a system acting on the positions.
- Make it possible to drop items from inventory and in general QoL features for inventory management.
- Item and Monster placement on the map should be more than just randomly trying to throw them in (not even caring if they are inside or outside the rooms).
- Coming up with new mechanics for mutations (I have some ideas already).

If anybody wants to follow my learning progress, the code is now available here: [GitHub](https://github.com/torlenor/asciiventure)

Please do not take that code as a guide how to do things, that's my first game and it is still a mess (but it's getting better!) ;)