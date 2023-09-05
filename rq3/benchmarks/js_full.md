This is the full benchmark set for JS of 25 cases

###### 1 ######
URL: https://github.com/marmelab/react-admin/pull/8057#discussion_r941544542

Review:

You forgot to update the JSDoc


Old Version:
/**
 * This hook returns a boolean indicating whether the form is invalid.
 * We use this to display an error message on submit in Form and SaveButton.
 *
 * We can't do the form validity check in the form submit handler
 * as the form state may not have been updated yet when onSubmit validation mode is enabled
 * or when the form hasn't been touched at all.
 */
export const useNotifyIsFormInvalid = (control?: Control) => {
    const { submitCount, errors } = useFormState(
        control ? { control } : undefined
    );
    const submitCountRef = useRef(submitCount);
    const notify = useNotify();

    useEffect(() => {
        // Checking the submit count allows us to only display the notification after users
        // tried to submit
        if (submitCount > submitCountRef.current) {
            submitCountRef.current = submitCount;

            if (Object.keys(errors).length > 0) {
                notify('ra.message.invalid_form', { type: 'warning' });
            }
        }
    }, [errors, submitCount, notify]);
};
New Version:
/**
 * This hook display an error message on submit in Form and SaveButton.
 * We use this to display an error message on submit in Form and SaveButton.
 *
 * We can't do the form validity check in the form submit handler
 * as the form state may not have been updated yet when onSubmit validation mode is enabled
 * or when the form hasn't been touched at all.
 */
export const useNotifyIsFormInvalid = (control?: Control) => {
    const { submitCount, errors } = useFormState(
        control ? { control } : undefined
    );
    const submitCountRef = useRef(submitCount);
    const notify = useNotify();

    useEffect(() => {
        // Checking the submit count allows us to only display the notification after users
        // tried to submit
        if (submitCount > submitCountRef.current) {
            submitCountRef.current = submitCount;

            if (Object.keys(errors).length > 0) {
                notify('ra.message.invalid_form', { type: 'warning' });
            }
        }
    }, [errors, submitCount, notify]);
};

###### 2 ######
URL: https://github.com/mochajs/mocha/pull/3632#discussion_r244403642

Review:Let's fix the JSDoc while we're here...

Old Version:
/**
 * Initialize a new `Suite` with the given `title` and `ctx`. Derived from [EventEmitter](https://nodejs.org/api/events.html#events_class_eventemitter)
 *
 * @memberof Mocha
 * @public
 * @class
 * @param {string} title
 * @param {Context} parentContext
 */
function Suite(title, parentContext) {
  if (!utils.isString(title)) {
    throw createInvalidArgumentTypeError(
      'Suite argument "title" must be a string. Received type "' +
        typeof title +
        '"',
      'title',
      'string'
    );
  }
  this.title = title;
  function Context() {}
  Context.prototype = parentContext;
  this.ctx = new Context();
  this.suites = [];
  this.tests = [];
  this.pending = false;
  this._beforeEach = [];
  this._beforeAll = [];
  this._afterEach = [];
  this._afterAll = [];
  this.root = !title;
  this._timeout = 2000;
  this._enableTimeouts = true;
  this._slow = 75;
  this._bail = false;
  this._retries = -1;
  this._onlyTests = [];
  this._onlySuites = [];
  this.delayed = false;
}
New Version:
/**
 * Constructs a new `Suite` instance with the given `title`, `ctx`, and `isRoot`.
 *
 * @public
 * @class
 * @extends EventEmitter
 * @memberof Mocha
 * @see {@link https://nodejs.org/api/events.html#events_class_eventemitter|EventEmitter}
 * @param {string} title - Suite title.
 * @param {Context} parentContext - Parent context instance.
 * @param {boolean} [isRoot=false] - Whether this is the root suite.
 */
function Suite(title, parentContext, isRoot) {
  if (!utils.isString(title)) {
    throw createInvalidArgumentTypeError(
      'Suite argument "title" must be a string. Received type "' +
        typeof title +
        '"',
      'title',
      'string'
    );
  }
  this.title = title;
  function Context() {}
  Context.prototype = parentContext;
  this.ctx = new Context();
  this.suites = [];
  this.tests = [];
  this.pending = false;
  this._beforeEach = [];
  this._beforeAll = [];
  this._afterEach = [];
  this._afterAll = [];
  this.root = isRoot === true;
  this._timeout = 2000;
  this._enableTimeouts = true;
  this._slow = 75;
  this._bail = false;
  this._retries = -1;
  this._onlyTests = [];
  this._onlySuites = [];
  this.delayed = false;
}

###### 3 ######
URL: https://github.com/spring-cloud/spring-cloud-dataflow-ui/commit/a9619f5c5bdc
Review: gh-432 Port field-value-counter D3js widgets to Angular4 infrastructure

* Migrate Pie chart component to Angular 4 and D3v4
  - Add ability to define # of pie slices
  - Add boolean flag wether to use all data to calculate`other` slice
* Migrate Bubble chart component to Angular 4 and D3v4
* Ensure that charts scale-up/down with browser-size
* Add model objects
* Integration charts into Dashboard
* Add service calls to `AnalyticsService`
* Lay ground-work to also support Aggregate Counters
* Fix Y-Axis issue for graph-chart (use `yScale(0)` not just `0` for value)
* Polish Bar-graph
  - Add basic update + transition-capabilities (Animations)
  - Add ability to specify the number of bars and width of bars via the Dashboard UI
* Add JsDoc
* Add Tests
* Add domain classes to support changing of the time resolution for aggregate counters
* Polish the UX around the dependent select-boxes on the Dashboard page
* Fixes #432
Old version:
/**
 * Retrieves all counters. Will take pagination into account.
 *
 * @param detailed If true will request additional counter values from the REST endpoint
 */
private getSingleCounter(counterName: string): Observable<Counter> {
    const requestOptionsArgs: RequestOptionsArgs = HttpUtils.getDefaultRequestOptions();
    return this.http.get(this.metricsCountersUrl + '/' + counterName, requestOptionsArgs)
                    .map(response => {
                      const body = response.json();
                      console.log('body', body);
                      return new Counter().deserialize(body);
                    })
                    .catch(this.errorHandler.handleError);
}
New version:
/**
 * Retrieves a single counter.
 *
 * @param counterName Name of the counter for which to retrieve details
 */
private getSingleCounter(counterName: string): Observable<Counter> {
        const requestOptionsArgs: RequestOptionsArgs = HttpUtils.getDefaultRequestOptions();
        return this.http.get(this.metricsCountersUrl + '/' + counterName, requestOptionsArgs)
                        .map(response => {
                          const body = response.json();
                          console.log('body', body);
                          return new Counter().deserialize(body);
                        })
                        .catch(this.errorHandler.handleError);
}

###### 4 ######
URL: https://github.com/vitejs/vite-plugin-react-pages/commit/8018ab00c246
Review: chore: fix comment
Old version:
/**
 * If the page page does not come from local file,
 * then we create this api to track the source of page data.
 * When a file is unlinked,
 * the data associated with it will automatically get deleted.
 */
createAPIForSourceFile(
  sourceFile: File,
  scheduleUpdate: ScheduleUpdate
): HandlerAPI {
  const getRuntimeData: HandlerAPI['getRuntimeData'] = (pageId) =>
    this.createMutableProxy(pageId, 'runtime', sourceFile, scheduleUpdate)

  const getStaticData: HandlerAPI['getStaticData'] = (pageId) =>
    this.createMutableProxy(pageId, 'static', sourceFile, scheduleUpdate)

  const addPageData: HandlerAPI['addPageData'] = (pageData) => {
    const key = pageData.key ?? 'main'
    if (pageData.dataPath) {
      const runtimeData = getRuntimeData(pageData.pageId)
      runtimeData[key] = pageData.dataPath
    }
    if (pageData.staticData) {
      const staticData = getStaticData(pageData.pageId)
      staticData[key] = pageData.staticData
    }
  }

  return {
    getRuntimeData,
    getStaticData,
    addPageData,
  }
}
New version:
/**
 * If the page comes from local file,
 * then we create this api to track the source of page data.
 * When a file is unlinked,
 * the data associated with it will automatically get deleted.
 */
createAPIForSourceFile(
  sourceFile: File,
  scheduleUpdate: ScheduleUpdate
): HandlerAPI {
  const getRuntimeData: HandlerAPI['getRuntimeData'] = (pageId) =>
    this.createMutableProxy(pageId, 'runtime', sourceFile, scheduleUpdate)

  const getStaticData: HandlerAPI['getStaticData'] = (pageId) =>
    this.createMutableProxy(pageId, 'static', sourceFile, scheduleUpdate)

  const addPageData: HandlerAPI['addPageData'] = (pageData) => {
    const key = pageData.key ?? 'main'
    if (pageData.dataPath) {
      const runtimeData = getRuntimeData(pageData.pageId)
      runtimeData[key] = pageData.dataPath
    }
    if (pageData.staticData) {
      const staticData = getStaticData(pageData.pageId)
      staticData[key] = pageData.staticData
    }
  }

  return {
    getRuntimeData,
    getStaticData,
    addPageData,
  }
}

###### 5 ######
URL: https://github.com/0xs34n/starknet.js/commit/34a977953c2b
Review: fix(account): function documentation fix
Old version:
/**
 * Verify a signature of a given hash
 * @warning This method is not recommended, use verifyMessage instead
 *
 * @param hash - hash to be verified
 * @param signature - signature of the hash
 * @returns true if the signature is valid, false otherwise
 * @throws {Error} if the signature is not a valid signature
 */
public async verifyMessage(typedData: TypedData, signature: Signature): Promise<boolean> {
  const hash = await this.hashMessage(typedData);
  return this.verifyMessageHash(hash, signature);
}
New version:
/**
 * Verify a signature of a JSON object
 *
 * @param hash - hash to be verified
 * @param signature - signature of the hash
 * @returns true if the signature is valid, false otherwise
 * @throws {Error} if the signature is not a valid signature
 */
public async verifyMessage(typedData: TypedData, signature: Signature): Promise<boolean> {
  const hash = await this.hashMessage(typedData);
  return this.verifyMessageHash(hash, signature);
}


###### 6 ######
URL: https://github.com/AudiusProject/audius-protocol/pull/2186#discussion_r779951401

Review: docstring update

Old Version:
  /**
   * Given map(replica set node => userWallets[]), retrieves clock values for every (node, userWallet) pair
   * @param {Object} replicaSetNodesToUserWalletsMap map of <replica set node : wallets>
   * @param {Set<string>} unhealthyPeers set of unhealthy peer endpoints
   * @param {number?} [maxUserClockFetchAttempts=10] max number of attempts to fetch clock values
   *
   * @returns {Object} map of peer endpoints to (map of user wallet strings to clock value of replica set node for user)
   */
  async retrieveUserInfoFromReplicaSet(
    replicasToWalletsMap,
    unhealthyPeers,
    maxUserClockFetchAttempts = 10
  ) {
    const replicasToUserInfoMap = {}

    // TODO change to batched parallel requests
    const replicas = Object.keys(replicasToWalletsMap)
    await Promise.all(
      replicas.map(async (replica) => {
        replicasToUserInfoMap[replica] = {}

        const walletsOnReplica = replicasToWalletsMap[replica]

        const axiosReqParams = {
          baseURL: replica,
          url: '/users/batch_clock_status',
          method: 'post',
          data: { walletPublicKeys: walletsOnReplica },
          timeout: BATCH_CLOCK_STATUS_REQUEST_TIMEOUT
        }

        // Generate and attach SP signature to bypass route rate limits
        const { timestamp, signature } = generateTimestampAndSignature(
          { spID: this.spID },
          this.delegatePrivateKey
        )
        axiosReqParams.params = { spID: this.spID, timestamp, signature }

        // Make axios request with retries
        let userClockValuesResp = []
        let userClockFetchAttempts = 0
        let errorMsg
        while (userClockFetchAttempts++ < maxUserClockFetchAttempts) {
          try {
            userClockValuesResp = (await axios(axiosReqParams)).data.data.users
          } catch (e) {
            errorMsg = e
          }
        }
        // If failed to get response after all attempts, add replica to `unhealthyPeers` list for reconfig
        if (userClockValuesResp.length === 0) {
          this.logError(
            `[retrieveUserInfoFromReplicaSet] Could not fetch clock values for wallets=${walletsOnReplica} on replica node=${replica} ${
              errorMsg ? ': ' + errorMsg.toString() : ''
            }`
          )
          unhealthyPeers.add(replica)
        }

        // Else, add response data to output aggregate map
        userClockValuesResp.forEach((userClockValueResp) => {
          // If node is running behind version 0.3.50, `filesHash` value will be undefined
          const { walletPublicKey, clock, filesHash } = userClockValueResp
          replicasToUserInfoMap[replica][walletPublicKey] = { clock, filesHash }
        })
      })
    )

    return replicasToUserInfoMap
  }

New Version:
  /**
   * Given map(replica set node => userWallets[]), retrieves user info for every (node, userWallet) pair
   * @param {Object} replicaSetNodesToUserWalletsMap map of <replica set node : wallets>
   * @param {Set<string>} unhealthyPeers set of unhealthy peer endpoints
   * @param {number?} [maxUserClockFetchAttempts=10] max number of attempts to fetch clock values
   *
   * @returns {Object} map(replica => map(wallet => { clock, filesHash }))
   */
  async retrieveUserInfoFromReplicaSet(
    replicasToWalletsMap,
    unhealthyPeers,
    maxUserClockFetchAttempts = 10
  ) {
    const replicasToUserInfoMap = {}

    // TODO change to batched parallel requests
    const replicas = Object.keys(replicasToWalletsMap)
    await Promise.all(
      replicas.map(async (replica) => {
        replicasToUserInfoMap[replica] = {}

        const walletsOnReplica = replicasToWalletsMap[replica]

        const axiosReqParams = {
          baseURL: replica,
          url: '/users/batch_clock_status',
          method: 'post',
          data: { walletPublicKeys: walletsOnReplica },
          timeout: BATCH_CLOCK_STATUS_REQUEST_TIMEOUT
        }

        // Generate and attach SP signature to bypass route rate limits
        const { timestamp, signature } = generateTimestampAndSignature(
          { spID: this.spID },
          this.delegatePrivateKey
        )
        axiosReqParams.params = { spID: this.spID, timestamp, signature }

        // Make axios request with retries
        let userClockValuesResp = []
        let userClockFetchAttempts = 0
        let errorMsg
        while (userClockFetchAttempts++ < maxUserClockFetchAttempts) {
          try {
            userClockValuesResp = (await axios(axiosReqParams)).data.data.users
          } catch (e) {
            errorMsg = e
          }
        }
        // If failed to get response after all attempts, add replica to `unhealthyPeers` list for reconfig
        if (userClockValuesResp.length === 0) {
          this.logError(
            `[retrieveUserInfoFromReplicaSet] Could not fetch clock values for wallets=${walletsOnReplica} on replica node=${replica} ${
              errorMsg ? ': ' + errorMsg.toString() : ''
            }`
          )
          unhealthyPeers.add(replica)
        }

        // Else, add response data to output aggregate map
        userClockValuesResp.forEach((userClockValueResp) => {
          // If node is running behind version 0.3.50, `filesHash` value will be undefined
          const { walletPublicKey, clock, filesHash } = userClockValueResp
          replicasToUserInfoMap[replica][walletPublicKey] = { clock, filesHash }
        })
      })
    )

    return replicasToUserInfoMap
  }


###### 7 ######
URL: https://github.com/1Password/connect-sdk-js/commit/d7681a6094c6
Review: fixed pr comments
Old version:
/**
 * Get metadata about a single vault
 *
 * @param vaultId
 */
public async getVaultById(vaultId: string): Promise<Vault> {
    const { data } = await this.adapter.sendRequest(
        "get",
        `${this.basePath}/${vaultId}`,
    );
    return ObjectSerializer.deserialize(data, "Vault");
}
New version:
/**
 * Get metadata about a single vault with the provided ID.
 *
 * @param {string} vaultId
 */
public async getVaultById(vaultId: string): Promise<Vault> {
    const { data } = await this.adapter.sendRequest(
        "get",
        `${this.basePath}/${vaultId}`,
    );
    return ObjectSerializer.deserialize(data, "Vault");
}

###### 8 ######
URL: https://github.com/marmelab/react-admin/pull/8057#discussion_r941544542

Review:
You forgot to update the JSDoc

Old Version:
/**
 * This hook returns a boolean indicating whether the form is invalid.
 * We use this to display an error message on submit in Form and SaveButton.
 *
 * We can't do the form validity check in the form submit handler
 * as the form state may not have been updated yet when onSubmit validation mode is enabled
 * or when the form hasn't been touched at all.
 */
export const useNotifyIsFormInvalid = (control?: Control) => {
    const { submitCount, errors } = useFormState(
        control ? { control } : undefined
    );
    const submitCountRef = useRef(submitCount);
    const notify = useNotify();

    useEffect(() => {
        // Checking the submit count allows us to only display the notification after users
        // tried to submit
        if (submitCount > submitCountRef.current) {
            submitCountRef.current = submitCount;

            if (Object.keys(errors).length > 0) {
                notify('ra.message.invalid_form', { type: 'warning' });
            }
        }
    }, [errors, submitCount, notify]);
};

New Version:
/**
 * This hook display an error message on submit in Form and SaveButton.
 *
 * We can't do the form validity check in the form submit handler
 * as the form state may not have been updated yet when onSubmit validation mode is enabled
 * or when the form hasn't been touched at all.
 */
export const useNotifyIsFormInvalid = (control?: Control) => {
    const { submitCount, errors } = useFormState(
        control ? { control } : undefined
    );
    const submitCountRef = useRef(submitCount);
    const notify = useNotify();

    useEffect(() => {
        // Checking the submit count allows us to only display the notification after users
        // tried to submit
        if (submitCount > submitCountRef.current) {
            submitCountRef.current = submitCount;

            if (Object.keys(errors).length > 0) {
                notify('ra.message.invalid_form', { type: 'warning' });
            }
        }
    }, [errors, submitCount, notify]);
};

###### 9 ######
URL: https://github.com/scratchfoundation/scratch-vm/pull/1947#discussion_r257304691

Review:
This comment should reflect that this is a pre-processing step to start retrieving and loading assets and that watchers will be ignored... (because they don't have costumes and sounds...)

A side effect of these changes is that we could load extra assets from malformed project jsons (e.g. there's nothing preventing a watcher from having costumes and sounds). I would prefer to have an early exit in here similar to the one at the top of parseScratchObject...


Old Version:
/**
 * Parse a single "Scratch object" and create all its in-memory VM objects.
 * TODO: parse the "info" section, especially "savedExtensions"
 * @param {!object} object - From-JSON "Scratch object:" sprite, stage, watcher.
 * @param {!Runtime} runtime - Runtime object to load all structures into.
 * @param {ImportedExtensionsInfo} extensions - (in/out) parsed extension information will be stored here.
 * @param {boolean} topLevel - Whether this is the top-level object (stage).
 * @param {?object} zip - Optional zipped assets for local file import
 * @return {!Promise.<Array.<Target>>} Promise for the loaded targets when ready, or null for unsupported objects.
 */
const parseScratchAssets = function (object, runtime, topLevel, zip) {
    if (!object.hasOwnProperty('objName')) {
        // Skip parsing monitors. Or any other objects missing objName.
        return null;
    }

    const assets = {costumePromises: [], soundPromises: [], children: []};

    // Costumes from JSON.
    const costumePromises = assets.costumePromises;
    if (object.hasOwnProperty('costumes')) {
        for (let i = 0; i < object.costumes.length; i++) {
            const costumeSource = object.costumes[i];
            const bitmapResolution = costumeSource.bitmapResolution || 1;
            const costume = {
                name: costumeSource.costumeName,
                bitmapResolution: bitmapResolution,
                rotationCenterX: topLevel ? 240 * bitmapResolution : costumeSource.rotationCenterX,
                rotationCenterY: topLevel ? 180 * bitmapResolution : costumeSource.rotationCenterY,
                // TODO we eventually want this next property to be called
                // md5ext to reflect what it actually contains, however this
                // will be a very extensive change across many repositories
                // and should be done carefully and altogether
                md5: costumeSource.baseLayerMD5,
                skinId: null
            };
            const md5ext = costumeSource.baseLayerMD5;
            const idParts = StringUtil.splitFirst(md5ext, '.');
            const md5 = idParts[0];
            let ext;
            if (idParts.length === 2 && idParts[1]) {
                ext = idParts[1];
            } else {
                // Default to 'png' if baseLayerMD5 is not formatted correctly
                ext = 'png';
                // Fix costume md5 for later
                costume.md5 = `${costume.md5}.${ext}`;
            }
            costume.dataFormat = ext;
            costume.assetId = md5;
            if (costumeSource.textLayerMD5) {
                costume.textLayerMD5 = StringUtil.splitFirst(costumeSource.textLayerMD5, '.')[0];
            }
            // If there is no internet connection, or if the asset is not in storage
            // for some reason, and we are doing a local .sb2 import, (e.g. zip is provided)
            // the file name of the costume should be the baseLayerID followed by the file ext
            const assetFileName = `${costumeSource.baseLayerID}.${ext}`;
            const textLayerFileName = costumeSource.textLayerID ? `${costumeSource.textLayerID}.png` : null;
            costumePromises.push(deserializeCostume(costume, runtime, zip, assetFileName, textLayerFileName)
                .then(() => loadCostume(costume.md5, costume, runtime, 2 /* optVersion */))
            );
        }
    }
    // Sounds from JSON
    const soundPromises = assets.soundPromises;
    if (object.hasOwnProperty('sounds')) {
        for (let s = 0; s < object.sounds.length; s++) {
            const soundSource = object.sounds[s];
            const sound = {
                name: soundSource.soundName,
                format: soundSource.format,
                rate: soundSource.rate,
                sampleCount: soundSource.sampleCount,
                // TODO we eventually want this next property to be called
                // md5ext to reflect what it actually contains, however this
                // will be a very extensive change across many repositories
                // and should be done carefully and altogether
                // (for example, the audio engine currently relies on this
                // property to be named 'md5')
                md5: soundSource.md5,
                data: null
            };
            const md5ext = soundSource.md5;
            const idParts = StringUtil.splitFirst(md5ext, '.');
            const md5 = idParts[0];
            const ext = idParts[1].toLowerCase();
            sound.dataFormat = ext;
            sound.assetId = md5;
            // If there is no internet connection, or if the asset is not in storage
            // for some reason, and we are doing a local .sb2 import, (e.g. zip is provided)
            // the file name of the sound should be the soundID (provided from the project.json)
            // followed by the file ext
            const assetFileName = `${soundSource.soundID}.${ext}`;
            soundPromises.push(deserializeSound(sound, runtime, zip, assetFileName).then(() => sound));
        }
    }

    // The stage will have child objects; recursively process them.
    const childrenAssets = assets.children;
    if (object.children) {
        for (let m = 0; m < object.children.length; m++) {
            childrenAssets.push(parseScratchAssets(object.children[m], runtime, false, zip));
        }
    }

    return assets;
};

New Version:
/**
 * Parse the assets of a single "Scratch object" and load them. This
 * preprocesses objects to support loading the data for those assets over a
 * network while the objects are further processed into Blocks, Sprites, and a
 * list of needed Extensions.
 * @param {!object} object - From-JSON "Scratch object:" sprite, stage, watcher.
 * @param {!Runtime} runtime - Runtime object to load all structures into.
 * @param {boolean} topLevel - Whether this is the top-level object (stage).
 * @param {?object} zip - Optional zipped assets for local file import
 * @return {?{costumePromises:Array.<Promise>,soundPromises:Array.<Promise>,children:object}}
 *   Object of arrays of promises and child objects for asset objects used in
 *   Sprites. null for unsupported objects.
 */
const parseScratchAssets = function (object, runtime, topLevel, zip) {
    if (!object.hasOwnProperty('objName')) {
        // Skip parsing monitors. Or any other objects missing objName.
        return null;
    }

    const assets = {costumePromises: [], soundPromises: [], children: []};

    // Costumes from JSON.
    const costumePromises = assets.costumePromises;
    if (object.hasOwnProperty('costumes')) {
        for (let i = 0; i < object.costumes.length; i++) {
            const costumeSource = object.costumes[i];
            const bitmapResolution = costumeSource.bitmapResolution || 1;
            const costume = {
                name: costumeSource.costumeName,
                bitmapResolution: bitmapResolution,
                rotationCenterX: topLevel ? 240 * bitmapResolution : costumeSource.rotationCenterX,
                rotationCenterY: topLevel ? 180 * bitmapResolution : costumeSource.rotationCenterY,
                // TODO we eventually want this next property to be called
                // md5ext to reflect what it actually contains, however this
                // will be a very extensive change across many repositories
                // and should be done carefully and altogether
                md5: costumeSource.baseLayerMD5,
                skinId: null
            };
            const md5ext = costumeSource.baseLayerMD5;
            const idParts = StringUtil.splitFirst(md5ext, '.');
            const md5 = idParts[0];
            let ext;
            if (idParts.length === 2 && idParts[1]) {
                ext = idParts[1];
            } else {
                // Default to 'png' if baseLayerMD5 is not formatted correctly
                ext = 'png';
                // Fix costume md5 for later
                costume.md5 = `${costume.md5}.${ext}`;
            }
            costume.dataFormat = ext;
            costume.assetId = md5;
            if (costumeSource.textLayerMD5) {
                costume.textLayerMD5 = StringUtil.splitFirst(costumeSource.textLayerMD5, '.')[0];
            }
            // If there is no internet connection, or if the asset is not in storage
            // for some reason, and we are doing a local .sb2 import, (e.g. zip is provided)
            // the file name of the costume should be the baseLayerID followed by the file ext
            const assetFileName = `${costumeSource.baseLayerID}.${ext}`;
            const textLayerFileName = costumeSource.textLayerID ? `${costumeSource.textLayerID}.png` : null;
            costumePromises.push(deserializeCostume(costume, runtime, zip, assetFileName, textLayerFileName)
                .then(() => loadCostume(costume.md5, costume, runtime, 2 /* optVersion */))
            );
        }
    }
    // Sounds from JSON
    const soundPromises = assets.soundPromises;
    if (object.hasOwnProperty('sounds')) {
        for (let s = 0; s < object.sounds.length; s++) {
            const soundSource = object.sounds[s];
            const sound = {
                name: soundSource.soundName,
                format: soundSource.format,
                rate: soundSource.rate,
                sampleCount: soundSource.sampleCount,
                // TODO we eventually want this next property to be called
                // md5ext to reflect what it actually contains, however this
                // will be a very extensive change across many repositories
                // and should be done carefully and altogether
                // (for example, the audio engine currently relies on this
                // property to be named 'md5')
                md5: soundSource.md5,
                data: null
            };
            const md5ext = soundSource.md5;
            const idParts = StringUtil.splitFirst(md5ext, '.');
            const md5 = idParts[0];
            const ext = idParts[1].toLowerCase();
            sound.dataFormat = ext;
            sound.assetId = md5;
            // If there is no internet connection, or if the asset is not in storage
            // for some reason, and we are doing a local .sb2 import, (e.g. zip is provided)
            // the file name of the sound should be the soundID (provided from the project.json)
            // followed by the file ext
            const assetFileName = `${soundSource.soundID}.${ext}`;
            soundPromises.push(deserializeSound(sound, runtime, zip, assetFileName).then(() => sound));
        }
    }

    // The stage will have child objects; recursively process them.
    const childrenAssets = assets.children;
    if (object.children) {
        for (let m = 0; m < object.children.length; m++) {
            childrenAssets.push(parseScratchAssets(object.children[m], runtime, false, zip));
        }
    }

    return assets;
};
###### 10 ######
URL: https://github.com/ing-bank/lion/pull/281#discussion_r328484505

Review:
Yes, we should change the comment to something like "Dispatch submit event and invoke submit on the native form when clicked"

Old Version:
  /**
   * Prevent normal click and redispatch click on the native button unless already redispatched.
   */
  __clickDelegationHandler() {
    if (this.type === 'submit' && this._nativeButtonNode.form) {
      this._nativeButtonNode.form.dispatchEvent(new Event('submit'));
      this._nativeButtonNode.form.submit();
    }
  }

New Version:
  /**
   * Dispatch submit event and invoke submit on the native form when clicked
   */
  __clickDelegationHandler() {
    if (this.type === 'submit' && this._nativeButtonNode.form) {
      this._nativeButtonNode.form.dispatchEvent(new Event('submit'));
      this._nativeButtonNode.form.submit();
    }
  }

###### 11 ######
URL: https://github.com/symbol/symbol-sdk-typescript-javascript/commit/654b1638ec36

Review:
Fixed #117, Typo in AddressAliasTransaction and MosaicAliasTransaction comments

Old Version:
    /**
     * Create a mosaic supply change transaction object
     * @param deadline - The deadline to include the transaction.
     * @param actionType - The namespace id.
     * @param namespaceId - The namespace id.
     * @param mosaicId - The mosaic id.
     * @param networkType - The network type.
     * @param maxFee - (Optional) Max fee defined by the sender
     * @returns {AddressAliasTransaction}
    */
    public static create(deadline: Deadline,
                         actionType: AliasActionType,
                         namespaceId: NamespaceId,
                         address: Address,
                         networkType: NetworkType,
                         maxFee: UInt64 = new UInt64([0, 0])): AddressAliasTransaction {
        return new AddressAliasTransaction(networkType,
            TransactionVersion.ADDRESS_ALIAS,
            deadline,
            maxFee,
            actionType,
            namespaceId,
            address,
        );
}
New Version:
    /**
     * Create a address alias transaction object
     * @param deadline - The deadline to include the transaction.
     * @param actionType - The alias action type.
     * @param namespaceId - The namespace id.
     * @param address - The address.
     * @param networkType - The network type.
     * @param maxFee - (Optional) Max fee defined by the sender
     * @returns {AddressAliasTransaction}
     */
    public static create(deadline: Deadline,
                         actionType: AliasActionType,
                         namespaceId: NamespaceId,
                         address: Address,
                         networkType: NetworkType,
                         maxFee: UInt64 = new UInt64([0, 0])): AddressAliasTransaction {
        return new AddressAliasTransaction(networkType,
            TransactionVersion.ADDRESS_ALIAS,
            deadline,
            maxFee,
            actionType,
            namespaceId,
            address,
        );
}

###### 12 ######
URL: https://github.com/webrtc/testrtc/commit/daa7509f35c4
Review: Fix comments (CR feedback)
Old version:
// Sets |codec| as the default audio codec if it's present.
// The format of |codec| is 'NAME/RATE', e.g. 'opus/48000'.
function maybePreferCodec(sdp, type, dir, codec) {
  var str = type + ' ' + dir + ' codec';
  if (codec === '') {
    trace('No preference on ' + str + '.');
    return sdp;
  }

  trace('Prefer ' + str + ': ' + codec);

  var sdpLines = sdp.split('\r\n');

  // Search for m line.
  var mLineIndex = findLine(sdpLines, 'm=', type);
  if (mLineIndex === null) {
    return sdp;
  }

  // If the codec is available, set it as the default in m line.
  var codecIndex = findLine(sdpLines, 'a=rtpmap', codec);
  if (codecIndex) {
    var payload = getCodecPayloadType(sdpLines[codecIndex]);
    if (payload) {
      sdpLines[mLineIndex] = setDefaultCodec(sdpLines[mLineIndex], payload);
    }
  }

  sdp = sdpLines.join('\r\n');
  return sdp;
}
New version:
// Sets |codec| as the default |type| codec if it's present.
// The format of |codec| is 'NAME/RATE', e.g. 'opus/48000'.
function maybePreferCodec(sdp, type, dir, codec) {
  var str = type + ' ' + dir + ' codec';
  if (codec === '') {
    trace('No preference on ' + str + '.');
    return sdp;
  }

  trace('Prefer ' + str + ': ' + codec);

  var sdpLines = sdp.split('\r\n');

  // Search for m line.
  var mLineIndex = findLine(sdpLines, 'm=', type);
  if (mLineIndex === null) {
    return sdp;
  }

  // If the codec is available, set it as the default in m line.
  var codecIndex = findLine(sdpLines, 'a=rtpmap', codec);
  if (codecIndex) {
    var payload = getCodecPayloadType(sdpLines[codecIndex]);
    if (payload) {
      sdpLines[mLineIndex] = setDefaultCodec(sdpLines[mLineIndex], payload);
    }
  }

  sdp = sdpLines.join('\r\n');
  return sdp;
}
###### 13 ######
URL: https://github.com/tldr-pages/tldr-node-client/commit/e62978058849
Review: index.js: various documentation fixes
Old version:
// Set the variable to null
function clearRuntimeIndex() {
  shortIndex = null;
}
New version:
// Set the shortIndex variable to null.
function clearRuntimeIndex() {
  shortIndex = null;
}
###### 14 ######
URL: https://github.com/googleapis/nodejs-firestore/commit/9cc94f44e1b5
Review: Fixing API documentation for 'id' (#163)
Old version:
/**
 * The last path document of the referenced document.
 *
 * @type {string}
 * @name DocumentReference#id
 * @readonly
 *
 * @example
 * let collectionRef = firestore.collection('col');
 *
 * collectionRef.add({foo: 'bar'}).then(documentReference => {
 *   console.log(`Added document with name '${documentReference.id}'`);
 * });
 */
get id() {
  return this._referencePath.id;
}
New version:
/**
 * The last path element of the referenced document.
 *
 * @type {string}
 * @name DocumentReference#id
 * @readonly
 *
 * @example
 * let collectionRef = firestore.collection('col');
 *
 * collectionRef.add({foo: 'bar'}).then(documentReference => {
 *   console.log(`Added document with name '${documentReference.id}'`);
 * });
 */
get id() {
  return this._referencePath.id;
}
###### 15 ######
URL: https://github.com/tronprotocol/tronweb/commit/f4f8bf8b461d
Review: Fixed comments and some detail in the function
Old version:
/**
 * Lists all network modification proposals.
 */
getAccountResources(address = false, callback = false) {
    if(!callback)
        return this.injectPromise(this.getAccountResources, address);

    if(!this.tronWeb.isAddress(address))
        return callback('Invalid address provided');

    this.tronWeb.fullNode.request('wallet/getaccountresource', { 
        address: this.tronWeb.address.toHex(address),
    }, 'post').then(resources => {
        callback(null, resources);
    }).catch(err => callback(err));
}
New version:
/**
 * Get the account resources
 */
getAccountResources(address = false, callback = false) {
    if(!callback)
        return this.injectPromise(this.getAccountResources, address);

    if(!this.tronWeb.isAddress(address))
        return callback('Invalid address provided');

    this.tronWeb.fullNode.request('wallet/getaccountresource', { 
        address: this.tronWeb.address.toHex(address),
    }, 'post').then(resources => {
        callback(null, resources);
    }).catch(err => callback(err));
}

###### 16 ######
URL: https://github.com/harmonyland/harmony/commit/fe6930c06523
Review: miner grammer/typo fixes is comments. Changed all comments I can find to consider method as a singular noun, tho I might missed a few comments
Old version:
/** Check if every value/key in Collection satisfy callback */
every(callback: (value: V, key: K) => boolean): boolean {
  for (const key of this.keys()) {
    const value = this.get(key) as V
    if (!callback(value, key)) return false
  }
  return true
}
New version:
/** Check if every value/key in Collection satisfies callback */
every(callback: (value: V, key: K) => boolean): boolean {
  for (const key of this.keys()) {
    const value = this.get(key) as V
    if (!callback(value, key)) return false
  }
  return true
}

###### 17 ######
URL: https://github.com/tj/commander.js/commit/c3b419ac892f
Review: Fix comments to refer to argument rather than option (#1630)
Old version:
/**
 * Make option-argument optional.
 */
argOptional() {
  this.required = false;
  return this;
}
New version:
/**
 * Make argument optional.
 */
argOptional() {
  this.required = false;
  return this;
}
###### 18 ######
URL: https://github.com/wavesurfer-js/wavesurfer.js/commit/45144cf7d4a0
Review: doc: add missing and fix invalid jsdoc statements (#1647)

* add missing and fix invalid jsdoc

* improve jsdoc
Old version:
/**
 * Set the playback volume.
 *
 * @param {string} deviceId String value representing underlying output device
 */
setSinkId(deviceId) {
    return this.backend.setSinkId(deviceId);
}
New version:
/**
 * Sets the ID of the audio device to use for output and returns a Promise.
 *
 * @param {string} deviceId String value representing underlying output
 * device
 * @returns {Promise} `Promise` that resolves to `undefined` when there are
 * no errors detected.
 */
setSinkId(deviceId) {
    return this.backend.setSinkId(deviceId);
}
###### 19 ######
URL: https://github.com/KaTeX/KaTeX/commit/f63af87f17fe
Review: Add looots of comments

Summary:
Add comments everywhere! Also fix some small bugs like using Style.id
instead of Style.size, and rename some variables to be more descriptive.

Fixes #22

Test Plan:
 - Make sure the huxley screenshots didn't change
 - Make sure the tests still pass

Reviewers: alpert

Reviewed By: alpert

Differential Revision: http://phabricator.khanacademy.org/D13158
Old version:
// The result of a single lex
function LexResult(type, text, position) {
    this.type = type;
    this.text = text;
    this.position = position;
}
New version:
// The resulting token returned from `lex`.
function LexResult(type, text, position) {
    this.type = type;
    this.text = text;
    this.position = position;
}
###### 20 ######
URL: https://github.com/vuex-orm/vuex-orm/commit/c3a1345277ca
Review: Fix comments
Old version:
/**
 * Creates module from given models.
 */
static create (entities: Entity[]): Vuex.Module<any, any> {
  return {
    namespaced: true,
    modules: this.createTree(entities)
  }
}
New version:
/**
 * Creates module from the given entities.
 */
static create (entities: Entity[]): Vuex.Module<any, any> {
  return {
    namespaced: true,
    modules: this.createTree(entities)
  }
}
###### 21 ######
URL: https://github.com/rokucommunity/brighterscript/commit/11667047f5b8
Review: Fix comment typo
Old version:
/**
 * Find a list of files in the program that have a function with the given name (case INsensitive)
 */
public findFilesForClass(className: string) {
    const files = [] as BscFile[];
    const lowerClassName = className.toLowerCase();
    //find every file with this class defined
    for (const file of Object.values(this.files)) {
        if (isBrsFile(file)) {
            //TODO handle namespace-relative classes
            //if the file has a function with this name
            if (file.parser.references.classStatementLookup.get(lowerClassName) !== undefined) {
                files.push(file);
            }
        }
    }
    return files;
}
New version:
/**
 * Find a list of files in the program that have a class with the given name (case INsensitive)
 */
public findFilesForClass(className: string) {
    const files = [] as BscFile[];
    const lowerClassName = className.toLowerCase();
    //find every file with this class defined
    for (const file of Object.values(this.files)) {
        if (isBrsFile(file)) {
            //TODO handle namespace-relative classes
            //if the file has a function with this name
            if (file.parser.references.classStatementLookup.get(lowerClassName) !== undefined) {
                files.push(file);
            }
        }
    }
    return files;
}

###### 22 ######
URL: https://github.com/aws-quickstart/cdk-eks-blueprints/commit/912468c2e017
Review: fix code comments
Old version:
/**
 * Setup the secrets for CSI driver
 * @param clusterInfo
 */
setupSecrets(clusterInfo: ClusterInfo, team: ApplicationTeam, csiDriver: Construct): void {
  // Create the service account for the team
  this.addPolicyToServiceAccount(clusterInfo, team);

  // Create and apply SecretProviderClass manifest
  this.createSecretProviderClass(clusterInfo, team, csiDriver);
}
New version:
/**
 * Setup Team secrets
 * @param clusterInfo 
 * @param team 
 * @param csiDriver 
 */
setupSecrets(clusterInfo: ClusterInfo, team: ApplicationTeam, csiDriver: Construct): void {
  // Create the service account for the team
  this.addPolicyToServiceAccount(clusterInfo, team);

  // Create and apply SecretProviderClass manifest
  this.createSecretProviderClass(clusterInfo, team, csiDriver);
}

###### 23 ######
URL: https://github.com/microsoft/vscode-js-profile-visualizer/commit/9010c204c15e
Review: chore: fix comment
Old version:
/**
 * Adds a new code lens at the given location in the file.
 */
protected set(file: string, position: Position, data: ITreeNode) {
  let list = this.data.get(lowerCaseInsensitivePath(file));
  if (!list) {
    list = [];
    this.data.set(lowerCaseInsensitivePath(file), list);
  }

  let index = 0;
  while (index < list.length && list[index].position.line < position.line) {
    index++;
  }

  if (list[index]?.position.line === position.line) {
    const existing = list[index];
    if (position.character < existing.position.character) {
      existing.position = new Position(position.line, position.character);
    }
    existing.data.totalSize += data.totalSize;
    existing.data.selfSize += data.selfSize;
  } else {
    list.splice(index, 0, {
      position: new Position(position.line, position.character),
      data: {
        totalSize: data.totalSize,
        selfSize: data.selfSize,
      },
    });
  }
}
New version:
/**
 * Adds a new code lens at the given treeNode in the file.
 */
protected set(file: string, position: Position, data: ITreeNode) {
  let list = this.data.get(lowerCaseInsensitivePath(file));
  if (!list) {
    list = [];
    this.data.set(lowerCaseInsensitivePath(file), list);
  }

  let index = 0;
  while (index < list.length && list[index].position.line < position.line) {
    index++;
  }

  if (list[index]?.position.line === position.line) {
    const existing = list[index];
    if (position.character < existing.position.character) {
      existing.position = new Position(position.line, position.character);
    }
    existing.data.totalSize += data.totalSize;
    existing.data.selfSize += data.selfSize;
  } else {
    list.splice(index, 0, {
      position: new Position(position.line, position.character),
      data: {
        totalSize: data.totalSize,
        selfSize: data.selfSize,
      },
    });
  }
}

###### 24 ######
URL: https://github.com/vuex-orm/vuex-orm-next/commit/bf9b63b61e4c
Review: chore: fix typo in the comment
Old version:
/**
 * Commit `deleteAll` change to the store.
 */
function flush(state: State): void {
  state.data = {}
}
New version:
/**
 * Commit `flush` change to the store.
 */
function flush(state: State): void {
  state.data = {}
}

###### 25 ######
URL: https://github.com/lit/lit-element/commit/9a9aced52111
Review: Documentation fixes based on feedback.
Old version:
/**
 * Validates the element by updating it via `update`, `finishUpdate`,
 * and `finishFirstUpdate`.
 */
private _validate() {
  // Mixin instance properties once, if they exist.
  if (this._instanceProperties) {
    this._applyInstanceProperties();
  }
  if (this.shouldUpdate(this._changedProperties)) {
    this.update(this._changedProperties);
  } else {
    this._markUpdated();
  }
}
New version:
/**
 * Validates the element by updating it via `update`.
 */
private _validate() {
  // Mixin instance properties once, if they exist.
  if (this._instanceProperties) {
    this._applyInstanceProperties();
  }
  if (this.shouldUpdate(this._changedProperties)) {
    this.update(this._changedProperties);
  } else {
    this._markUpdated();
  }
}
