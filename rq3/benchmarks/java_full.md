This is the full benchmark set for Java of 25 cases

###### 1 ######
URL: https://github.com/microsoft/gctoolkit/pull/204/files/ea10831a1d3a5b0ed42d7a523aa249df9f6c9992#r797024552

Review: language doesn't quite make sense here?

Old Version:
    /**
     * of the first event is significantly away from zero in relation to the time intervals between the
     * of the next N events, where N maybe 1.
     *
     * try to estimate the time at which the JVM started. For log fragments, this will be the time
     * of the first event in the log. Otherwise it will be 0.000 seconds.
     * @return DateTimeStamp
     */
    @Override
    public DateTimeStamp getEstimatedJVMStartTime() {
        DateTimeStamp startTime = getTimeOfFirstEvent();
        // Initial entries in GC log happen within seconds. Lets allow for 60 before considering the log
        // to be a fragment.
        if (startTime.getTimeStamp() < LOG_FRAGMENT_THRESHOLD) {
            return startTime.minus(startTime.getTimeStamp());
        } else {
            return startTime;
        }
    }


New Version:
    /**
     * If the first event is significantly distant from zero in relation to the time intervals between the
     * of the next N events, where N maybe 1, then this is likely a log fragment and not the start of the run.
     * <p>
     * Try to estimate the time at which the JVM started. For log fragments, this will be the time
     * of the first event in the log. Otherwise it will be 0.000 seconds.
     *
     * @return DateTimeStamp as estimated start time.
     */
    @Override
    public DateTimeStamp getEstimatedJVMStartTime() {
        DateTimeStamp startTime = getTimeOfFirstEvent();
        // Initial entries in GC log happen within seconds. Lets allow for 60 before considering the log
        // to be a fragment.
        if (startTime.getTimeStamp() < LOG_FRAGMENT_THRESHOLD_SECONDS) {
            return startTime.minus(startTime.getTimeStamp());
        } else {
            return startTime;
        }
    }


###### 2 ######
URL: https://github.com/Azure/azure-sdk-for-android/pull/244/files/feb0690a4e64d58dea4b38cba26a2968820cb75d#r441213254

Review:
Javadoc need to be fixed I think.

Old Version:
    /**
     * Reads the blob's metadata & properties.
     *
     * @param containerName The container name.
     * @param blobName      The blob name.
     * @param callback      Callback that receives the response.
     */
    ServiceCall delete(String containerName,
                       String blobName,
                       Callback<Void> callback) {
        return storageBlobServiceClient.delete(containerName,
            blobName,
            callback);
    }

New Version:
    /**
     * Deletes the specified blob or snapshot. Note that deleting a blob also deletes all its snapshots.
     *
     * @param containerName The container name.
     * @param blobName      The blob name.
     * @param callback      Callback that receives the response.
     * @return A handle to the service call.
     */
    ServiceCall delete(String containerName,
                       String blobName,
                       Callback<Void> callback) {
        return storageBlobServiceClient.delete(containerName,
            blobName,
            callback);
    }


###### 3 ######
URL: https://github.com/spring-projects/spring-amqp/pull/318#discussion_r36679052

Review:
s/Add/Remove and so on.
In other words: the wrong JavaDocs.
Can be fixed on merge if that's all.

Old Version:
	/**
	 * Add the mapping of a queue name or consumer tag to a method name. The first lookup
	 * is by queue name, if that returns null, we lookup by consumer tag, if that
	 * returns null, the {@link #setDefaultListenerMethod(String) defaultListenerMethod}
	 * is used.
	 * @param queueOrTag The queue name or consumer tag.
	 * @return the method name that was removed, or null.
	 * @since 1.5
	 */
	public String removeQueueOrTagToMethodName(String queueOrTag) {
		return this.queueOrTagToMethodName.remove(queueOrTag);
	}

New Version:
	/**
	 * Remove the mapping of a queue name or consumer tag to a method name.
	 * @param queueOrTag The queue name or consumer tag.
	 * @return the method name that was removed, or null.
	 * @since 1.5
	 */
	public String removeQueueOrTagToMethodName(String queueOrTag) {
		return this.queueOrTagToMethodName.remove(queueOrTag);
	}

###### 4 ######
URL: https://github.com/AdoptOpenJDK/IcedTea-Web/pull/228#discussion_r283750540

Review:
you are right, Javadoc does not explicitly say this (only in the @return text). I've improved doc for this. It was also my first reaction when I saw the code that I wanted the method ti return false here. But after studying the existing test cases it looks like this behavior was by intention.  And I didn't want to change functionality here with this refactoring. So for now I would keep it like this until we have more confidence where to check for side effects when making this method more \"logic\".

Old Version:
    /**
     * Checks whether the first part of the given prefixString is a prefix for any of the strings
     * in the specified array.
     * If the {@code prefixString} contains multiple words separated by a space character, the
     * first word is taken as prefix for comparison.
     *
     * @param prefixString the prefixString string
     * @param available the strings to test
     * @return true if the first part of the given prefixString is a prefix for any of the strings
     * in the specified array or the specified array is empty or null, false otherwise
     */
    public static boolean hasPrefixMatch(final String prefixString, final String[] available) {
        Assert.requireNonBlank(prefixString, "prefixString");

        if (available == null || available.length == 0){
            return true;
        }

        for (final String candidate : available) {
            final String trimmedPrefix = prefixString.split(WHITESPACE_CHARACTER_SEQUENCE)[0];
            String trimmedCandidate = null;
            if (candidate != null) {
                trimmedCandidate = candidate.split(WHITESPACE_CHARACTER_SEQUENCE)[0];
            }
            if (trimmedCandidate != null && trimmedCandidate.startsWith(trimmedPrefix)) {
                return true;
            }
        }

        return false;
    }
New Version:
    /**
     * Checks whether the first part of the given prefixString is a prefix for any of the strings
     * in the specified array. If no array is specified (empty or null) it is considered to be a
     * match.
     *
     * If the {@code prefixString} contains multiple words separated by a space character, the
     * first word is taken as prefix for comparison.
     *
     * @param prefixString the prefixString string
     * @param available the strings to test
     * @return true if the first part of the given prefixString is a prefix for any of the strings
     * in the specified array or the specified array is empty or null, false otherwise
     */
    public static boolean hasPrefixMatch(final String prefixString, final String[] available) {
        Assert.requireNonBlank(prefixString, "prefixString");

        if (available == null || available.length == 0){
            return true;
        }

        final String trimmedPrefix = prefixString.split(WHITESPACE_CHARACTER_SEQUENCE)[0];

        for (final String candidate : available) {
            String trimmedCandidate = null;
            if (candidate != null) {
                trimmedCandidate = candidate.split(WHITESPACE_CHARACTER_SEQUENCE)[0];
            }
            if (trimmedCandidate != null && trimmedCandidate.startsWith(trimmedPrefix)) {
                return true;
            }
        }

        return false;
    }

###### 5 ######
URL: https://github.com/eclipse/reddeer/pull/2006#discussion_r277978536

Review: Can you please update javadoc comment for all methods for setting modifier? e.g. from Sets a given modifier. to Sets modifier to public.

Old Version:
    /**
	 * Sets a given modifier.
	 * 
	 */
	public NewInterfaceCreationWizardPage setPublicModifier() {
		new RadioButton(this, "public").toggle(true);
		return this;
	}



New Version:
    /**
	 * Sets modifier to public.
	 * 
	 */
	public NewInterfaceCreationWizardPage setPublicModifier() {
		new RadioButton(this, "public").toggle(true);
		return this;
	}



###### 6 ######
URL: https://github.com/gridgain/gridgain/pull/1107#discussion_r428773531

Review: Javadoc is outdated.

Old Version:
        /**
         * @return List of indexes names.
         */
        public String indexesRegEx() {
            return indexesRegEx;
        }

New Version:
        /**
         * @return Index names regex filter.
         */
        public String indexesRegEx() {
            return indexesRegEx;
        }

###### 7 ######
URL: https://github.com/onthegomap/planetiler/pull/463#discussion_r1105596308

Review: Opened #486 to fix the javadoc.

Old Version:
 /** Returns the raw tile data associated with the tile at {@code coord}. */
  default byte[] getTile(TileCoord coord) {
    return getTile(coord.x(), coord.y(), coord.z());
  }

New Version:
  /** Returns the raw tile data at {@code coord} or {@code null} if not found. */
  default byte[] getTile(TileCoord coord) {
    return getTile(coord.x(), coord.y(), coord.z());
  }

###### 8 ######
URL: https://github.com/mesosphere/dcos-commons/pull/470#discussion_r97919940

Review: Lowercase is legal but unusual. I've only seen lowercase in shell scripts for internal use (ie unexported variables). Updated the javadoc to clarify.

Old Version:
/**
     * Converts the provided string to a valid environment variable name.
     *
     * For example: {@code hello.There!} => {@code HELLO_THERE_}
     */
    public static String toEnvName(String str) {
        return ENVVAR_INVALID_CHARS.matcher(str.toUpperCase()).replaceAll("_");
}

New Version:
/**
     * Converts the provided string to a conventional environment variable name, consisting of numbers, uppercase
     * letters, and underscores. Strictly speaking, lowercase characters are not invalid, but this avoids them to follow
     * convention.
     *
     * For example: {@code hello.There999!} => {@code HELLO_THERE999_}
     */
    public static String toEnvName(String str) {
        return ENVVAR_INVALID_CHARS.matcher(str.toUpperCase()).replaceAll("_");
}

###### 9 ######
URL: https://github.com/spring-projects/spring-integration/pull/189#discussion_r244962

Review: This javadoc is no longer correct since the introspection was refactored out. It needs to be updated.

Old Version:
/**
	 * Adds the outbound or inbound prefix if necessary.
	 */
	private String addPrefixIfNecessary(String prefix, String propertyName) {
		String headerName = propertyName;
		if (StringUtils.hasText(prefix) && !headerName.startsWith(prefix)) {
			headerName = prefix + propertyName;
		}
		return headerName;
	}

New Version:
/**
	 * Adds the prefix to the header name
	 */
	private String addPrefixIfNecessary(String prefix, String propertyName) {
		String headerName = propertyName;
		if (StringUtils.hasText(prefix) && !headerName.startsWith(prefix)) {
			headerName = prefix + propertyName;
		}
		return headerName;
	}

###### 10 ######
URL: https://github.com/line/armeria/pull/1797#discussion_r288409404

Review: I'm wondering is this method supposed to wrap or copy or be deleted. The javadoc doesn't mention the potential danger of wrapping like `HttpData` javadoc always has. Should we update the javadoc?

Old Version:
    /**
     * Creates a new HTTP request.
     *
     * @param method the HTTP method of the request
     * @param path the path of the request
     * @param mediaType the {@link MediaType} of the request content
     * @param content the content of the request
     */
    static AggregatedHttpRequest of(HttpMethod method, String path, MediaType mediaType, byte[] content) {
        requireNonNull(method, "method");
        requireNonNull(path, "path");
        requireNonNull(mediaType, "mediaType");
        requireNonNull(content, "content");
        return of(method, path, mediaType, HttpData.wrap(content));
    }

New Version:
    /**
     * Creates a new HTTP request. The {@code content} will be wrapped using {@link HttpData#wrap(byte[])}, so
     * any changes made to {@code content} will be reflected in the request.
     *
     * @param method the HTTP method of the request
     * @param path the path of the request
     * @param mediaType the {@link MediaType} of the request content
     * @param content the content of the request
     */
    static AggregatedHttpRequest of(HttpMethod method, String path, MediaType mediaType, byte[] content) {
        requireNonNull(method, "method");
        requireNonNull(path, "path");
        requireNonNull(mediaType, "mediaType");
        requireNonNull(content, "content");
        return of(method, path, mediaType, HttpData.wrap(content));
    }

###### 11 ######
URL: https://github.com/dhis2/dhis2-core/pull/12392#discussion_r1048303118

Review: I think the comment here is misleading? This is about a filter parameter, not about the flag of the tei itself - correct?

Old Version:
    /**
     * Indicates whether this parameters specifies if tei is a potential
     * duplicate. It can be true or false.
     */
    public boolean hasPotentialDuplicate()
    {
        return potentialDuplicate != null;
    }

New Version:
    /**
     * Indicates whether we are filtering for potential duplicate.
     */
    public boolean hasPotentialDuplicate()
    {
        return potentialDuplicate != null;
    }

###### 12 ######
URL: https://github.com/f2prateek/dart/pull/39#discussion_r34426632

Review:
First comment : I didn't really understand the difference with the one above.\n\nUpdate: Yes I did. I would change the first javadoc : `Returns a bundler that wraps a new Bundle, all data of {@code source} bundle is copied into the new bundle.`\n

Old Version:
  /** Returns a bundler that delegates to the source bundle. */
  public static Bundler of(Bundle source) {
    return new Bundler(source);
  }

New Version:
  /** Returns a bundler that wraps a new Bundle, all data of {@code source} bundle is copied into the new bundle. */
  public static Bundler of(Bundle source) {
    return new Bundler(source);
  }

###### 13 ######
URL: https://github.com/greenplum-db/pxf/pull/542#discussion_r573486621

Review: 

need to update Javadoc here and for other changed constructors


Old Version:
    /**
     * Empty Constructor
     */
    public GPDBWritable() {
        initializeEightByteAlignment();
    }

New Version:
    /**
     * Constructs a {@link GPDBWritable} object with a given
     * {@code databaseEncoding}
     */
    public GPDBWritable(Charset databaseEncoding) {
        this.databaseEncoding = databaseEncoding;
        initializeEightByteAlignment();
    }

###### 14 ######
URL: https://github.com/linkedin/avro-util/pull/428/files#r1049077891

Review: this doesnt traverse into records. updated the javadoc

Old Version:
  /**
   * checks if the value for a given schema can possibly contain
   * strings. this is important when dealing with things like Utf8 vs java.lang.Strings
   * @param schema a schema
   * @return true if value under schema could possibly involve strings
   */
  public static boolean schemaContainsString(Schema schema) {
    if (schema == null) {
      return false;
    }
    boolean hasString = false;
    switch (schema.getType()) {
      case STRING:
      case MAP: //map keys are always strings, regardless of values
        return true;
      case UNION:
        // Any member can have string?
        for(Schema branch : schema.getTypes()) {
          if (schemaContainsString(branch)) {
            return true;
          }
        }
        return false;
      case ARRAY:
        return schemaContainsString(schema.getElementType());
    }

    return false;
  }

New Version:
  /**
   * checks if the value for a given schema can possibly contain
   * strings (meaning is a string, union containing string, or collections
   * containing any of the above).
   * this is important when dealing with things like Utf8 vs java.lang.Strings
   * @param schema a schema
   * @return true if value under schema could possibly involve strings
   */
  public static boolean schemaContainsString(Schema schema) {
    if (schema == null) {
      return false;
    }
    boolean hasString = false;
    switch (schema.getType()) {
      case STRING:
      case MAP: //map keys are always strings, regardless of values
        return true;
      case UNION:
        // Any member can have string?
        for(Schema branch : schema.getTypes()) {
          if (schemaContainsString(branch)) {
            return true;
          }
        }
        return false;
      case ARRAY:
        return schemaContainsString(schema.getElementType());
    }

    return false;
  }

###### 15 ######
URL: https://github.com/spring-projects/spring-authorization-server/pull/1056#discussion_r1116862441

Review: Please update the javadoc as the passwordencoder is not used to validate the clientSecret but instead used to encode the clientSecret.\r\nAlso, please add `@since 1.1.0`

Old Version:
/**
	 * Sets the {@link PasswordEncoder} used to validate the
	 * the {@link RegisteredClient#getClientSecret() client secret}.
	 * If not set, the client secret will be encoded using
	 * {@link PasswordEncoderFactories#createDelegatingPasswordEncoder()}.
	 *
	 * @param passwordEncoder the {@link PasswordEncoder} used to encode the clientSecret
	 */
	public void setPasswordEncoder(PasswordEncoder passwordEncoder) {
		Assert.notNull(passwordEncoder, "passwordEncoder cannot be null");
		this.passwordEncoder = passwordEncoder;
	}

New Version:
/**
	 * Sets the {@link PasswordEncoder} used to encode the clientSecret
	 * the {@link RegisteredClient#getClientSecret() client secret}.
	 * If not set, the client secret will be encoded using
	 * {@link PasswordEncoderFactories#createDelegatingPasswordEncoder()}.
	 *
	 * @param passwordEncoder the {@link PasswordEncoder} used to encode the clientSecret
	 * @since 1.1.0
	 */
	public void setPasswordEncoder(PasswordEncoder passwordEncoder) {
		Assert.notNull(passwordEncoder, "passwordEncoder cannot be null");
		this.passwordEncoder = passwordEncoder;
	}

###### 16 ######
URL: https://github.com/apple/servicetalk/pull/1515#discussion_r619538821

Review: I updated the javadoc. ptal.

Old Version:
    /**
     * Create a new {@link Single} that emits the results of a specified zipper {@link BiFunction} to items emitted by
     * {@code singles}. All operations will terminate (even if there are failures) before the returned {@link Single}
     * terminates.
     * <p>
     * From a sequential programming point of view this method is roughly equivalent to the following:
     * <pre>{@code
     *      CompletableFuture<T> f1 = ...; // this
     *      CompletableFuture<T2> other = ...;
     *      CompletableFuture.allOf(f1, other).get(); // wait for all futures to complete
     *      return zipper.apply(f1.get(), other.get());
     * }</pre>
     * @param other The other {@link Single} to zip with.
     * @param zipper Used to combine the completed results for each item from {@code singles}.
     * @param <T2> The type of {@code other}.
     * @param <R> The result type of the zipper.
     * @return a new {@link Single} that emits the results of a specified zipper {@link BiFunction} to items emitted by
     * {@code singles}.
     * @see <a href="http://reactivex.io/documentation/operators/zip.html">ReactiveX zip operator.</a>
     */
    public final <T2, R> Single<R> zipWithDelayError(Single<? extends T2> other,
                                                     BiFunction<? super T, ? super T2, ? extends R> zipper) {
        return zipDelayError(this, other, zipper);
    }
New Version:
    /**
     * Create a new {@link Single} that emits the results of a specified zipper {@link BiFunction} to items emitted by
     * {@code this} and {@code other}. If any of the {@link Single}s terminate with an error, the returned
     * {@link Single} will wait for termination till all the other {@link Single}s have been subscribed and terminated,
     * and then terminate with the first error.
     * <p>
     * From a sequential programming point of view this method is roughly equivalent to the following:
     * <pre>{@code
     *      CompletableFuture<T> f1 = ...; // this
     *      CompletableFuture<T2> other = ...;
     *      CompletableFuture.allOf(f1, other).get(); // wait for all futures to complete
     *      return zipper.apply(f1.get(), other.get());
     * }</pre>
     * @param other The other {@link Single} to zip with.
     * @param zipper Used to combine the completed results for each item from {@code singles}.
     * @param <T2> The type of {@code other}.
     * @param <R> The result type of the zipper.
     * @return a new {@link Single} that emits the results of a specified zipper {@link BiFunction} to items emitted by
     * {@code this} and {@code other}.
     * @see <a href="http://reactivex.io/documentation/operators/zip.html">ReactiveX zip operator.</a>
     */
    public final <T2, R> Single<R> zipWithDelayError(Single<? extends T2> other,
                                                     BiFunction<? super T, ? super T2, ? extends R> zipper) {
        return zipDelayError(this, other, zipper);
    }


###### 17 ######
URL: https://github.com/nordic-institute/X-Road/pull/481#discussion_r419916273

Review:
We need our custom `CsrfValidationFilter` since the original `CsrfFilter` only verifies the one CsrfToken returned from `loadToken` here. So CsrfFilter only checks the value from **session attribute** and compares it to value from **request header**.\r\n\r\nIf we changed loadToken so that it \r\n- also loads value from **cookie**\r\n- compares it to value from session attribute\r\n- fails with `accessDeniedHandler.handle()` if those do not match\r\n\r\nThat way CsrfFilter + CookieAndSessionCsrfTokenRepository would be verifying that cookie == header == session attribute.\r\n\r\nThat way we could eliminate the custom CsrfValidationFilter and just use the original CsrFilter?\r\n\r\nIt is maybe a slight deviation of what you could expect from the players (repository stores and loads token, CsrfFilter validates token), but I did not see any hard rules in javadocs that we would be breaking, and it is also kind of logical; if repository stores tokens into two places, and loads only one value, it makes sense that it validates that value 1 == value 2 when loading one of the values.
Old Version:
    /**
     * The de facto token that gets loaded from the session
     */
    @Override
    public CsrfToken loadToken(HttpServletRequest request) {
        HttpSession session = request.getSession(false);
        if (session != null) {
            return (CsrfToken) session.getAttribute(this.sessionAttributeName);
        }
        return null;
    }
New Version:
    /**
     * Validate and load the token if there is a session. If there is no session -> return null
     */
    @Override
    public CsrfToken loadToken(HttpServletRequest request) {
        HttpSession session = request.getSession(false);
        // validate csrf only if a session exists
        if (session != null) {
            return validateAndLoadToken(request);
        }
        return null;
    }

###### 18 ######
URL: https://github.com/dnsjava/dnsjava/pull/56#discussion_r290069343

Review:I'm definitely fine with making this change to use Lists.\r\n\r\nI'm not going to stop a breaking change, given this is a major version bump, but another possible way forward without breaking old consumers is to introduce new method(s) with different names, `getRRs` or `getDataRecords` (or whatever you want to name them) that uses the new List return type.  Then add the @Deprecated annotation and javadoc tag to the old Iterator-based rss() methods so consumers know to use the new ones (and switch all internal code to use the new ones).  If you go that route, apply the same to `sigs` -> `getSigs()` and elsewhere.\r\n\r\nJust throwing that out there.

Old Version:
/**
 * Returns an Iterator listing all (data) records.
 * @param cycle If true, cycle through the records so that each Iterator will
 * start with a different record.
 */
public List<T>
rrs(boolean cycle) {
	if (!cycle || rrs.size() <= 1) {
		return Collections.unmodifiableList(rrs);
	}

	List<T> l = new ArrayList<>(rrs.size());
	int start = position++ % rrs.size();
	l.addAll(rrs.subList(start, rrs.size()));
	l.addAll(rrs.subList(0, start));
	return l;
}

New Version:
/**
 * Returns a list of all data records.
 * @param cycle If true, cycle through the records so that each list will
 * start with a different record.
 */
public List<T>
rrs(boolean cycle) {
	if (!cycle || rrs.size() <= 1) {
		return Collections.unmodifiableList(rrs);
	}

	List<T> l = new ArrayList<>(rrs.size());
	int start = position++ % rrs.size();
	l.addAll(rrs.subList(start, rrs.size()));
	l.addAll(rrs.subList(0, start));
	return l;
}

###### 19 ######
URL: https://github.com/openjdk/jdk/pull/2109#discussion_r559007774

Review: The fix looks okay but the javadoc might be a big clearer if you move the text into the method description.

Old Version:
    /**
     * @return Returns the address of the host represented by this URL.
     *         A {@link SecurityException} or an {@link UnknownHostException}
     *         while getting the host address will result in this method returning
     *         {@code null}
     */
    synchronized InetAddress getHostAddress() {
        if (hostAddress != null) {
            return hostAddress;
        }

        if (host == null || host.isEmpty()) {
            return null;
        }
        try {
            hostAddress = InetAddress.getByName(host);
        } catch (UnknownHostException | SecurityException ex) {
            return null;
        }
        return hostAddress;
    }

New Version:
    /**
     * Returns the address of the host represented by this URL.
     * A {@link SecurityException} or an {@link UnknownHostException}
     * while getting the host address will result in this method returning
     * {@code null}
     *
     * @return an {@link InetAddress} representing the host
     */
    synchronized InetAddress getHostAddress() {
        if (hostAddress != null) {
            return hostAddress;
        }
        if (host == null || host.isEmpty()) {
            return null;
        }
        try {
            hostAddress = InetAddress.getByName(host);
        } catch (UnknownHostException | SecurityException ex) {
            return null;
        }
        return hostAddress;
}

###### 20 ######
URL: https://github.com/fozziethebeat/S-Space/pull/1#discussion_r91364

Review:
Can you update the javadoc with the pathLength argument description. What if pathLength is negative?


Old Version:
    /**
     * Creates and configures this {@code DependencyVectorSpace} with the
     * according to the provided properties.  If no properties are specified,
     * the default values are used.
     */
    public DependencyVectorSpace(Properties properties, int pathLength) {
        if (pathLength < 0)
            throw new IllegalArgumentException(
                    "path length must be non-negative");

        termToVector = new HashMap<String,SparseDoubleVector>();

        String basisMappingProp = 
            properties.getProperty(BASIS_MAPPING_PROPERTY);
        basisMapping = (basisMappingProp == null)
            ? new WordBasedBasisMapping()
            : ReflectionUtil.<DependencyPathBasisMapping>
                getObjectInstance(basisMappingProp);
        String pathWeightProp = 
            properties.getProperty(PATH_WEIGHTING_PROPERTY);
        weighter = (pathWeightProp == null)
            ? new FlatPathWeight()
            : ReflectionUtil.<DependencyPathWeight>
                getObjectInstance(pathWeightProp);
        String acceptorProp = 
            properties.getProperty(PATH_ACCEPTOR_PROPERTY);
        acceptor = (acceptorProp == null)
            ? new MinimumPennTemplateAcceptor()
            : ReflectionUtil.<DependencyPathAcceptor>
                getObjectInstance(acceptorProp);

        this.pathLength = (pathLength == 0)
            ? acceptor.maxPathLength()
            : pathLength;

        extractor = DependencyExtractorManager.getDefaultExtractor();
    }

New Version:
    /**
     * Creates and configures this {@code DependencyVectorSpace} with the
     * default set of parameters.  The default values are:<ul>
     *   <li> a {@link WordBasedBasisMapping} is used for dimensions;
     *   <li> a {@link FlatPathWeight} is used to weight accepted paths;
     *   <li> and a {@link MinimumTemplateAcceptor} is used to filter the paths
     *        in a sentence.
     * </ul>
     *
     * @param properties The {@link Properties} setting the above options
     * @param pathLength The maximum valid path length.  Must be non-negative.
     *        If zero, an the maximum path length used by the {@link
     *        DependencyPathAcceptor} will be used.
     */
    public DependencyVectorSpace(Properties properties, int pathLength) {
        if (pathLength < 0)
            throw new IllegalArgumentException(
                    "path length must be non-negative");

        termToVector = new HashMap<String,SparseDoubleVector>();

        String basisMappingProp = 
            properties.getProperty(BASIS_MAPPING_PROPERTY);
        basisMapping = (basisMappingProp == null)
            ? new WordBasedBasisMapping()
            : ReflectionUtil.<DependencyPathBasisMapping>
                getObjectInstance(basisMappingProp);
        String pathWeightProp = 
            properties.getProperty(PATH_WEIGHTING_PROPERTY);
        weighter = (pathWeightProp == null)
            ? new FlatPathWeight()
            : ReflectionUtil.<DependencyPathWeight>
                getObjectInstance(pathWeightProp);
        String acceptorProp = 
            properties.getProperty(PATH_ACCEPTOR_PROPERTY);
        acceptor = (acceptorProp == null)
            ? new MinimumPennTemplateAcceptor()
            : ReflectionUtil.<DependencyPathAcceptor>
                getObjectInstance(acceptorProp);

        this.pathLength = (pathLength == 0)
            ? acceptor.maxPathLength()
            : pathLength;

        extractor = DependencyExtractorManager.getDefaultExtractor();
    }

###### 21 ######
URL: https://github.com/apache/ignite/pull/5656/files/a548b914eb64bb9914af7964a368791338d273ca#r302021407

Review:
Please update Javadoc to provide a good description. What does original distribution mean? For example, UUID - is it primary node id?
The Javadoc is the face of the product and it must be good enough for all users, especially if we are talking about public API.

Also, there is no test that covers this method.

Old Version:
    /**
     * Original distribution.
     */
    public Map<UUID, Map<Object, Object>> getEntries() {
        return locEntries;
    }

New Version:
    /**
     * Returns a mapping node ids to a collection of original entries affected by a cache operation.
     * @return Collection of original entries.
     */
    public Map<UUID, Map<K, V>> getEntries() {
        return originalEntries;
    }

###### 22 ######
URL: https://github.com/oracle/weblogic-kubernetes-operator/pull/923#discussion_r267435140

Review:
The Javadoc should describe the usecase more detail as follows  ...\r\n\r\n  Modify the Domain Scoped env property in the Domain Object using  kubectl apply -f domain.yaml \r\n  Make sure all the Server Pods in the domain got re-started \r\n  Reference https://github.com/oracle/weblogic-kubernetes-operator/blob/master/site/server-lifecycle.md  
Old Version:
  /**
   * The property tested is: env: "-Dweblogic.StdoutDebugEnabled=false"-->
   * "-Dweblogic.StdoutDebugEnabled=true"
   *
   * @throws Exception
   */
  @Test
  public void testServerPodsRestartByChangingEnvProperty() throws Exception {
    Assume.assumeFalse(QUICKTEST);
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    logTestBegin(testMethodName);

    boolean testCompletedSuccessfully = false;
    logger.info(
        "About to testDomainServerPodRestart for Domain: "
            + domain.getDomainUid()
            + "  env property: StdoutDebugEnabled=false to StdoutDebugEnabled=true");
    domain.testDomainServerPodRestart(
        "\"-Dweblogic.StdoutDebugEnabled=false\"", "\"-Dweblogic.StdoutDebugEnabled=true\"");

    logger.info("SUCCESS - " + testMethodName);
  }

New Version:
  /**
   * Modify the domain scope env property on the domain resource using kubectl apply -f domain.yaml
   * Verify that all the server pods in the domain got re-started. The property tested is: env:
   * "-Dweblogic.StdoutDebugEnabled=false"--> "-Dweblogic.StdoutDebugEnabled=true"
   *
   * @throws Exception
   */
  @Test
  public void testServerPodsRestartByChangingEnvProperty() throws Exception {
    Assume.assumeFalse(QUICKTEST);
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    logTestBegin(testMethodName);

    logger.info(
        "About to testDomainServerPodRestart for Domain: "
            + domain.getDomainUid()
            + "  env property: StdoutDebugEnabled=false to StdoutDebugEnabled=true");
    domain.testDomainServerPodRestart(
        "\"-Dweblogic.StdoutDebugEnabled=false\"", "\"-Dweblogic.StdoutDebugEnabled=true\"");

    logger.info("SUCCESS - " + testMethodName);
  }

###### 23 ######
URL: https://github.com/msemys/esjc/pull/44#discussion_r262459266

Review:
could you please update the javadocs

Old Version:
    /**
     * Sets whether or not to disconnect the client on detecting a channel error. By default, it is enabled and
     * client disconnects immediately. If it is disabled the client tries to reconnect according to {@link #maxReconnections(int)}.
     *
     * @param disconnectOnTcpChannelError {@code true} to disconnect or {@code false} to try to reconnect.
     * @return the builder reference
     */
    public EventStoreBuilder disconnectOnTcpChannelError(boolean disconnectOnTcpChannelError) {
        settingsBuilder.disconnectOnTcpChannelError(disconnectOnTcpChannelError);
        return this;
    }



New Version:
    /**
     * Sets whether or not to disconnect the client on detecting a channel error. By default, it is disabled and the client
     * tries to reconnect according to {@link #maxReconnections(int)}. If it is enabled the client disconnects immediately.
     *
     * @param disconnectOnTcpChannelError {@code true} to disconnect or {@code false} to try to reconnect.
     * @return the builder reference
     */
    public EventStoreBuilder disconnectOnTcpChannelError(boolean disconnectOnTcpChannelError) {
        settingsBuilder.disconnectOnTcpChannelError(disconnectOnTcpChannelError);
        return this;
    }

###### 24 ######
URL: https://github.com/apache/hbase/pull/493#discussion_r314153523

Review:
I think we need a careful javadoc here, say why we need the public method ... because exposing an MultiRowRangeFilter constructor with rowKeyPrefixes  looks very strange if no doc here.

Old Version:
  /**
   * @param rowKeyPrefixes the array of byte array
   */
  public MultiRowRangeFilter(byte[][] rowKeyPrefixes) {
    this(createRangeListFromRowKeyPrefixes(rowKeyPrefixes));
  }
New Version:
  /**
   * Constructor for creating a <code>MultiRowRangeFilter</code> from multiple rowkey prefixes.
   *
   * As <code>MultiRowRangeFilter</code> javadoc says (See the solution 1 of the first statement),
   * if you try to create a filter list that scans row keys corresponding to given prefixes (e.g.,
   * <code>FilterList</code> composed of multiple <code>PrefixFilter</code>s), this constructor
   * provides a way to avoid creating an inefficient one.
   *
   * @param rowKeyPrefixes the array of byte array
   */
  public MultiRowRangeFilter(byte[][] rowKeyPrefixes) {
    this(createRangeListFromRowKeyPrefixes(rowKeyPrefixes));
  }
###### 25 ######
URL: https://github.com/elastic/apm-agent-java/pull/1206/files/332e1fb2bd497411d6dbb32d5fdace4ee89d6e20#r439964581

Review:
Update javadoc
Old Version:
    /**
     * Wraps the provided runnable and makes this {@link AbstractSpan} active in the {@link Runnable#run()} method.
     *
     * <p>
     * Note: does activates the {@link AbstractSpan} and not only the {@link TraceContext}.
     * This should only be used when the span is closed in thread the provided {@link Runnable} is executed in.
     * </p>
     */
    @Nullable
    public static <T> Callable<T> withContext(@Nullable Callable<T> callable, @Nullable ElasticApmTracer tracer) {
        if (callable instanceof CallableLambdaWrapper || callable == null || tracer == null  || needsContext.get() == Boolean.FALSE) {
            return callable;
        }
        needsContext.set(Boolean.FALSE);
        AbstractSpan<?> active = tracer.getActive();
        if (active == null) {
            return callable;
        }
        if (isLambda(callable)) {
            callable = new CallableLambdaWrapper<>(callable);
        }
        ElasticApmAgent.ensureInstrumented(callable.getClass(), RUNNABLE_CALLABLE_FJTASK_INSTRUMENTATION);
        contextMap.put(callable, active);
        active.incrementReferences();
        return callable;
    }
New Version:
    /**
     * Instruments or wraps the provided runnable and makes this {@link AbstractSpan} active in the {@link Runnable#run()} method.
     */
    @Nullable
    public static <T> Callable<T> withContext(@Nullable Callable<T> callable, @Nullable ElasticApmTracer tracer) {
        if (callable instanceof CallableLambdaWrapper || callable == null || tracer == null  || needsContext.get() == Boolean.FALSE) {
            return callable;
        }
        needsContext.set(Boolean.FALSE);
        AbstractSpan<?> active = tracer.getActive();
        if (active == null) {
            return callable;
        }
        if (isLambda(callable)) {
            callable = new CallableLambdaWrapper<>(callable);
        }
        ElasticApmAgent.ensureInstrumented(callable.getClass(), RUNNABLE_CALLABLE_FJTASK_INSTRUMENTATION);
        contextMap.put(callable, active);
        active.incrementReferences();
        return callable;
    }