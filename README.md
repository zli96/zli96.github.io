# Hugo.io - Classic Theme

Classic is stylized fork of the **XMin** theme, written by [Yihui Xie](https://yihui.name).

Provides `highlight.js` for syntax highlighting, emoji support, Inter and optional **_darkmode_**.

[**View live demo**](https://goodroot.ca)

### Instructions

1: Install Hugo.

```
brew install hugo
```

2: Create a new site.

```
hugo new site classic
```

3: Change to themes dir.

```
cd classic/themes
```

4: Clone the repo

```
git clone git@github.com:goodroot/hugo-classic.git
```

5: Copy files within the `exampleSite` directory into the classic directory. Overwrite the existing `content/`, `static/`, and `config.toml` files.

6: Run `hugo server` within `classic/` and enjoy and customize to your hearts content!

### New Posts

To make new posts, simply use the command line:

```
hugo new post/good-to-great.md
```

### Header Colour

To adjust the header colour, head to `static/css/style.css` and change...

```
header {
    background: #613DC1;
}
```

... `background:` to any colour value you'd like!

For header font:

```
header a {
    color: #fff;
}
```

Change `color:` to a nice matching colour.

### Darkmode

1. Open `static/css/style.css`

2. Edit the following attributes to match light/dark

```css
/* darkmode */
@media (prefers-color-scheme: dark) {
    ...
}

/* lightmode */
@media (prefers-color-scheme: light) {
    ...
}
```

#### Screenshot

![Screenshot of Hugo Classic](/images/screenshot.png)

## Blog Posts

hugo-classic has appeared in...

[15 Hugo Framework blog themes](https://terrty.net/2018/15-hugo-framework-blog-themes/) by [paskal](https://github.com/paskal)
