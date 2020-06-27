---
layout: post
title:  "Lili's Quest | Week 5"
date:   2020-06-27 11:45:00 +0200
categories: "Lili's quest"
---
This week was heavily driven by refactoring. A lot more entity properties got moved into components, which only hold data. Currently the references to these components are still part of the entity, so for now the entity is still more than just an ID as one would usually have in a strict entity-component-system sense.

During that refactoring also the parsing of the entity definitions (JSON files) got much easier. Utilizing Go's awesome marshalling/unmarshalling interface the parse functions got shortened and types like MutationEffect now know themselves how to unmarshal from a JSON string into an actually MutationEffect.

In terms of game play I thought of some Mutations I want to add in a first iteration and which functionalities the game has to have for them. E.g., it would be nice to be able to dig through a wall, but for that I need to have destructible walls. In a similar way I want to have force fields, there I need to construct walls! 

Except of that I added a lot of TODOs. The next step will be to prioritize them. As game play should be my focus, I will probably start implementing the functionalities needed for the mutations first. UI will then be next on the list.