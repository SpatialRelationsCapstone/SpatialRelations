### Usage
- Install [yarn](https://yarnpkg.com/en/docs/install) (npm should work too).
- Run `yarn` or `yarn install` in the deploy folder to install dependencies.
- Run `yarn run build` quicker but larger development builds. Use `yarn run deploy` for smaller but slower builds. Use `yarn run (deploy/build) --watch` to rebuild automatically on changes to src.js.
- Host the dist folder as an HTTP server to avoid CORS errors. Something like `.../SpatialRelations/deploy/dist/$ python -m http.server`.

### TODO
- [ ] Make nodes squares with text
- [ ] Add attribute nodes
- [ ] Make predicates into nodes
- [ ] Add mouseover highlight to show relation between image and graph objects
- [ ] Make lines curvy?
- [ ] Everything else...