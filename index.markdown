---
layout: default
title: Blog
---

<h1>Posts</h1>

<ul>
  {% for post in site.posts %}
    <li>
      <h5>{{ post.date | date_to_string }}</h5>
      <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
    </li>
  {% endfor %}
</ul>
