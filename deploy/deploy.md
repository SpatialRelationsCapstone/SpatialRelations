### Usage
- Install [yarn](https://yarnpkg.com/en/docs/install), but npm should work too. Alternatively, just skip to the last step and use the prebuild bundle.js.
- Run `yarn` or `yarn install` in the deploy folder to install dependencies.
- Run `yarn run develop` for quick automatic rebuilds. Use `yarn run build` for a minified production build.
- Host the dist folder as an HTTP server to avoid CORS errors. Something like `.../SpatialRelations/deploy/dist/$ python -m http.server` should work.

### TODO
- [ ] Comment code
- [ ] Clean up code
- [ ] Make nodes squares with text
- [ ] Add attribute nodes
- [ ] Make predicates into nodes
- [ ] Add mouseover highlight to show relation between image and graph objects
- [ ] Ability to choose and upload image
- [ ] Add animation if model output is asynchronous
- [ ] Make lines curvy?
- [ ] Everything else...