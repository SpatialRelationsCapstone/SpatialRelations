### Usage
- Install [yarn](https://yarnpkg.com/en/docs/install), but npm should work too. Alternatively, just skip to the last step and use the prebuilt bundle.js.
- Run `yarn` or `yarn install` in the deploy folder to install dependencies.
- Run `yarn run develop` for quick automatic rebuilds. Use `yarn run build` for a minified production build.
- Host the dist folder as an HTTP server to avoid CORS errors. Can be done with python from the command line: `.../SpatialRelations/deploy/dist/$ python -m http.server`

### TODO
- [x] Comment code
- [ ] Clean up code
- [ ] Make nodes squares with text
- [ ] Add attribute nodes
- [x] Make predicates into nodes
- [ ] Add different styles for different link/node types
- [ ] Better looking in general
- [ ] Better physics params
- [ ] Draggable
- [ ] Add mouseover highlight to show relation between image and graph objects
- [ ] Ability to choose and upload image
- [ ] Add animation if model output is asynchronous
- [ ] Make lines curvy?
- [ ] Everything else...