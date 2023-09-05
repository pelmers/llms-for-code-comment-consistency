This is the full benchmark set for GO of 25 cases

###### 1 ######
URL: https://github.com/confluentinc/confluent-kafka-go/pull/937#discussion_r1095529182

Review:This comment needs to be updated to reflect changes

Old Version:
// testconf_read reads the test suite config file testconf.json which must
// contain at least Brokers and Topic string properties.
// Returns true if the testconf was found and usable, false if no such file, or panics
// if the file format is wrong.
func testconfRead() bool {
	cf, err := os.Open("testconf.json")
	if err != nil && !testconf.Docker {
		fmt.Fprintf(os.Stderr, "%% testconf.json not found and docker compose not setup - ignoring test\n")
		return false
	}

	// Default values
	testconf.PerfMsgCount = defaulttestconfPerfMsgCount
	testconf.PerfMsgSize = defaulttestconfPerfMsgSize
	testconf.GroupID = defaulttestconfGroupID
	testconf.Topic = defaulttestconfTopic
	testconf.Brokers = ""

	jp := json.NewDecoder(cf)
	err = jp.Decode(&testconf)
	if err != nil {
		panic(fmt.Sprintf("Failed to parse testconf: %s", err))
	}

	cf.Close()

	if testconf.Docker {
		testconf.Brokers = defaulttestconfBrokers
	}

	if !testconf.Docker && testconf.Brokers == "" {
		fmt.Fprintf(os.Stderr, "No Brokers provided in testconf")
		return false
	}

	if testconf.Brokers[0] == '$' {
		testconf.Brokers = os.Getenv(testconf.Brokers[1:])
	}

	return true
}
New Version:
// testconf_read reads the test suite config file testconf.json which must
// contain at least Brokers and Topic string properties or the defaults will be used.
// Returns true if the testconf was found and usable, false if no such file, or panics
// if the file format is wrong.
func testconfRead() bool {
	cf, err := os.Open("testconf.json")
	if err != nil && !testconf.Docker {
		fmt.Fprintf(os.Stderr, "%% testconf.json not found and docker compose not setup - ignoring test\n")
		return false
	}

	// Default values
	testconf.PerfMsgCount = defaulttestconfPerfMsgCount
	testconf.PerfMsgSize = defaulttestconfPerfMsgSize
	testconf.GroupID = defaulttestconfGroupID
	testconf.Topic = defaulttestconfTopic
	testconf.Brokers = ""

	jp := json.NewDecoder(cf)
	err = jp.Decode(&testconf)
	if err != nil {
		panic(fmt.Sprintf("Failed to parse testconf: %s", err))
	}

	cf.Close()

	if testconf.Docker {
		testconf.Brokers = defaulttestconfBrokers
	}

	if !testconf.Docker && testconf.Brokers == "" {
		fmt.Fprintf(os.Stderr, "No Brokers provided in testconf")
		return false
	}

	if testconf.Brokers[0] == '$' {
		testconf.Brokers = os.Getenv(testconf.Brokers[1:])
	}

	return true
}

###### 2 ######
URL: https://github.com/cilium/cilium/pull/2684#discussion_r165204442

Review:Update the documentation for this function to reflect the addition of `policy.GetPolicyEnabled()` having its result checked.

Old Version:
// IngressOrEgressIsEnforced returns true if either ingress or egress is in
// enforcement mode
func (e *Endpoint) IngressOrEgressIsEnforced() bool {
	return policy.GetPolicyEnabled() == AlwaysEnforce ||
		e.Opts.IsEnabled(OptionIngressPolicy) ||
		e.Opts.IsEnabled(OptionEgressPolicy)
}
New Version:
// IngressOrEgressIsEnforced returns true if either ingress or egress is in
// enforcement mode or if the global policy enforcement is enabled.
func (e *Endpoint) IngressOrEgressIsEnforced() bool {
	return policy.GetPolicyEnabled() == AlwaysEnforce ||
		e.Opts.IsEnabled(OptionIngressPolicy) ||
		e.Opts.IsEnabled(OptionEgressPolicy)
}

###### 3 ######
URL: https://github.com/viamrobotics/rdk/pull/498#discussion_r805000185

Review:update comment to GetPosition

Old Version:
// Position reports the position of the motor of the underlying robot
// based on its encoder. If it's not supported, the returned data is undefined.
// The unit returned is the number of revolutions which is intended to be fed
// back into calls of GoFor.
func (server *subtypeServer) GetPosition(
	ctx context.Context,
	req *pb.MotorServiceGetPositionRequest,
) (*pb.MotorServiceGetPositionResponse, error) {
	motorName := req.GetName()
	motor, err := server.getMotor(motorName)
	if err != nil {
		return nil, errors.Errorf("no motor (%s) found", motorName)
	}

	pos, err := motor.GetPosition(ctx)
	if err != nil {
		return nil, err
	}
	return &pb.MotorServiceGetPositionResponse{Position: pos}, nil
}
New Version:
// GetPosition reports the position of the motor of the underlying robot
// based on its encoder. If it's not supported, the returned data is undefined.
// The unit returned is the number of revolutions which is intended to be fed
// back into calls of GoFor.
func (server *subtypeServer) GetPosition(
	ctx context.Context,
	req *pb.MotorServiceGetPositionRequest,
) (*pb.MotorServiceGetPositionResponse, error) {
	motorName := req.GetName()
	motor, err := server.getMotor(motorName)
	if err != nil {
		return nil, errors.Errorf("no motor (%s) found", motorName)
	}

	pos, err := motor.GetPosition(ctx)
	if err != nil {
		return nil, err
	}
	return &pb.MotorServiceGetPositionResponse{Position: pos}, nil
}


###### 4 ######
URL: https://github.com/ipfs/go-mfs/pull/53#discussion_r246846256

Review:

I think Run's logic has changed enough to deserve an update in its documentation.


Old Version:
// Run contains the core logic of the `Republisher`. It calls the user-defined
// `pubfunc` function whenever the `Cid` value is updated. The complexity comes
// from the fact that `pubfunc` may be slow so we need to batch updates.
// Algorithm:
//   1. When we receive the first update after publishing, we set a `longer` timer.
//   2. When we receive any update, we reset the `quick` timer.
//   3. If either the `quick` timeout or the `longer` timeout elapses,
//      we call `publish` with the latest updated value.
//
// The `longer` timer ensures that we delay publishing by at most
// `TimeoutLong`. The `quick` timer allows us to publish sooner if
// it looks like there are no more updates coming down the pipe.
func (rp *Republisher) Run(lastPublished cid.Cid) {
	quick := time.NewTimer(0)
	if !quick.Stop() {
		<-quick.C
	}
	longer := time.NewTimer(0)
	if !longer.Stop() {
		<-longer.C
	}

	var toPublish cid.Cid
	for rp.ctx.Err() == nil {
		var waiter chan struct{}

		select {
		case <-rp.ctx.Done():
			return
		case newValue := <-rp.update:
			// Skip already published values.
			if lastPublished.Equals(newValue) {
				// Break to the end of the switch to cleanup any
				// timers.
				toPublish = cid.Undef
				break
			}

			// If we aren't already waiting to publish something,
			// reset the long timeout.
			if !toPublish.Defined() {
				longer.Reset(rp.TimeoutLong)
			}

			// Always reset the short timeout.
			quick.Reset(rp.TimeoutShort)

			// Finally, set the new value to publish.
			toPublish = newValue
			continue
		case waiter = <-rp.immediatePublish:
			// Make sure to grab the *latest* value to publish.
			select {
			case toPublish = <-rp.update:
			default:
			}

			// Avoid publishing duplicate values
			if !lastPublished.Equals(toPublish) {
				toPublish = cid.Undef
			}
		case <-quick.C:
		case <-longer.C:
		}

		// Cleanup, publish, and close waiters.

		// 1. Stop any timers. Don't use the `if !t.Stop() { ... }`
		//    idiom as these timers may not be running.

		quick.Stop()
		select {
		case <-quick.C:
		default:
		}

		longer.Stop()
		select {
		case <-longer.C:
		default:
		}

		// 2. If we have a value to publish, publish it now.
		if toPublish.Defined() {
			for {
				err := rp.pubfunc(rp.ctx, toPublish)
				if err == nil {
					break
				}
				// Keep retrying until we succeed or we abort.
				// TODO(steb): We could try pulling new values
				// off `update` but that's not critical (and
				// complicates this code a bit). We'll pull off
				// a new value on the next loop through.
				select {
				case <-time.After(rp.RetryTimeout):
				case <-rp.ctx.Done():
					return
				}
			}
			lastPublished = toPublish
			toPublish = cid.Undef
		}

		// 3. Trigger anything waiting in `WaitPub`.
		if waiter != nil {
			close(waiter)
		}
	}
}
New Version:
// Run contains the core logic of the `Republisher`. It calls the user-defined
// `pubfunc` function whenever the `Cid` value is updated to a *new* value. The
// complexity comes from the fact that `pubfunc` may be slow so we need to batch
// updates.
//
// Algorithm:
//   1. When we receive the first update after publishing, we set a `longer` timer.
//   2. When we receive any update, we reset the `quick` timer.
//   3. If either the `quick` timeout or the `longer` timeout elapses,
//      we call `publish` with the latest updated value.
//
// The `longer` timer ensures that we delay publishing by at most
// `TimeoutLong`. The `quick` timer allows us to publish sooner if
// it looks like there are no more updates coming down the pipe.
//
// Note: If a publish fails, we retry repeatedly every TimeoutRetry.
func (rp *Republisher) Run(lastPublished cid.Cid) {
	quick := time.NewTimer(0)
	if !quick.Stop() {
		<-quick.C
	}
	longer := time.NewTimer(0)
	if !longer.Stop() {
		<-longer.C
	}

	var toPublish cid.Cid
	for rp.ctx.Err() == nil {
		var waiter chan struct{}

		select {
		case <-rp.ctx.Done():
			return
		case newValue := <-rp.update:
			// Skip already published values.
			if lastPublished.Equals(newValue) {
				// Break to the end of the switch to cleanup any
				// timers.
				toPublish = cid.Undef
				break
			}

			// If we aren't already waiting to publish something,
			// reset the long timeout.
			if !toPublish.Defined() {
				longer.Reset(rp.TimeoutLong)
			}

			// Always reset the short timeout.
			quick.Reset(rp.TimeoutShort)

			// Finally, set the new value to publish.
			toPublish = newValue
			continue
		case waiter = <-rp.immediatePublish:
			// Make sure to grab the *latest* value to publish.
			select {
			case toPublish = <-rp.update:
			default:
			}

			// Avoid publishing duplicate values
			if !lastPublished.Equals(toPublish) {
				toPublish = cid.Undef
			}
		case <-quick.C:
		case <-longer.C:
		}

		// Cleanup, publish, and close waiters.

		// 1. Stop any timers. Don't use the `if !t.Stop() { ... }`
		//    idiom as these timers may not be running.

		quick.Stop()
		select {
		case <-quick.C:
		default:
		}

		longer.Stop()
		select {
		case <-longer.C:
		default:
		}

		// 2. If we have a value to publish, publish it now.
		if toPublish.Defined() {
			for {
				err := rp.pubfunc(rp.ctx, toPublish)
				if err == nil {
					break
				}
				// Keep retrying until we succeed or we abort.
				// TODO(steb): We could try pulling new values
				// off `update` but that's not critical (and
				// complicates this code a bit). We'll pull off
				// a new value on the next loop through.
				select {
				case <-time.After(rp.RetryTimeout):
				case <-rp.ctx.Done():
					return
				}
			}
			lastPublished = toPublish
			toPublish = cid.Undef
		}

		// 3. Trigger anything waiting in `WaitPub`.
		if waiter != nil {
			close(waiter)
		}
	}
}

###### 5 ######
URL: 
https://github.com/influxdata/influxdb/pull/11540#discussion_r250809484
Review:

Change the comments for this function; we're no longer using X-Influx-Error and X-Influx-Reference


Old Version:
// EncodeError encodes err with the appropriate status code and format,
// sets the X-Influx-Error and X-Influx-Reference headers on the response,
// and sets the response status to the corresponding status code.
func EncodeError(ctx context.Context, err error, w http.ResponseWriter) {
	if err == nil {
		return
	}

	code := platform.ErrorCode(err)
	httpCode, ok := statusCodePlatformError[code]
	if !ok {
		httpCode = http.StatusBadRequest
	}
	w.Header().Set(PlatformErrorCodeHeader, code)
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(httpCode)
	var e error
	if pe, ok := err.(*platform.Error); ok {
		e = &platform.Error{
			Code: code,
			Op:   platform.ErrorOp(err),
			Msg:  platform.ErrorMessage(err),
			Err:  pe.Err,
		}
	} else {
		e = &platform.Error{
			Code: platform.EInternal,
			Err:  err,
		}
	}
	b, _ := json.Marshal(e)
	_, _ = w.Write(b)
}
New Version:
// EncodeError encodes err with the appropriate status code and format,
// sets the X-Platform-Error-Code headers on the response.
// We're no longer using X-Influx-Error and X-Influx-Reference.
// and sets the response status to the corresponding status code.
func EncodeError(ctx context.Context, err error, w http.ResponseWriter) {
	if err == nil {
		return
	}

	code := platform.ErrorCode(err)
	httpCode, ok := statusCodePlatformError[code]
	if !ok {
		httpCode = http.StatusBadRequest
	}
	w.Header().Set(PlatformErrorCodeHeader, code)
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(httpCode)
	var e error
	if pe, ok := err.(*platform.Error); ok {
		e = &platform.Error{
			Code: code,
			Op:   platform.ErrorOp(err),
			Msg:  platform.ErrorMessage(err),
			Err:  pe.Err,
		}
	} else {
		e = &platform.Error{
			Code: platform.EInternal,
			Err:  err,
		}
	}
	b, _ := json.Marshal(e)
	_, _ = w.Write(b)
}

###### 6 ######
URL: https://github.com/grpc/grpc-go/pull/2681#discussion_r268913198

Review:
The comment seems to be outdated. e.g this function does not return anything.
Old Version:
// refreshSubConns creates/removes SubConns with backendAddrs. It returns a bool
// indicating whether the backendAddrs are different from the cached
// backendAddrs (whether any SubConn was newed/removed).
// Caller must hold lb.mu.
func (lb *lbBalancer) refreshSubConns(backendAddrs []resolver.Address, fromGRPCLBServer bool) {
	defer func() {
		// Regenerate and update picker after refreshing subconns because with
		// cache, even if SubConn was newed/removed, there might be no state
		// changes (the subconn will be kept in cache, not actually
		// newed/removed).
		lb.updateStateAndPicker(true, true)
	}()

	lb.inFallback = !fromGRPCLBServer

	opts := balancer.NewSubConnOptions{}
	if fromGRPCLBServer {
		opts.CredsBundle = lb.grpclbBackendCreds
	}
	lb.backendAddrs = nil
	if lb.usePickFirst {
		var sc balancer.SubConn
		for _, sc = range lb.subConns {
			break
		}
		if sc != nil {
			sc.UpdateAddresses(backendAddrs)
			sc.Connect()
			return
		}
		// This bypasses the cc wrapper with SubConn cache.
		sc, err := lb.cc.cc.NewSubConn(backendAddrs, opts)
		if err != nil {
			grpclog.Warningf("grpclb: failed to create new SubConn: %v", err)
			return
		}
		sc.Connect()
		lb.subConns[backendAddrs[0]] = sc
		lb.scStates[sc] = connectivity.Idle
		return
	}
	// addrsSet is the set converted from backendAddrs, it's used to quick
	// lookup for an address.
	addrsSet := make(map[resolver.Address]struct{})
	// Create new SubConns.
	for _, addr := range backendAddrs {
		addrWithoutMD := addr
		addrWithoutMD.Metadata = nil
		addrsSet[addrWithoutMD] = struct{}{}
		lb.backendAddrs = append(lb.backendAddrs, addrWithoutMD)
		if _, ok := lb.subConns[addrWithoutMD]; !ok {
			// Use addrWithMD to create the SubConn.
			sc, err := lb.cc.NewSubConn([]resolver.Address{addr}, opts)
			if err != nil {
				grpclog.Warningf("grpclb: failed to create new SubConn: %v", err)
				continue
			}
			lb.subConns[addrWithoutMD] = sc // Use the addr without MD as key for the map.
			if _, ok := lb.scStates[sc]; !ok {
				// Only set state of new sc to IDLE. The state could already be
				// READY for cached SubConns.
				lb.scStates[sc] = connectivity.Idle
			}
			sc.Connect()
		}
	}
	for a, sc := range lb.subConns {
		// a was removed by resolver.
		if _, ok := addrsSet[a]; !ok {
			lb.cc.RemoveSubConn(sc)
			delete(lb.subConns, a)
			// Keep the state of this sc in b.scStates until sc's state becomes Shutdown.
			// The entry will be deleted in HandleSubConnStateChange.
		}
	}
}
New Version:
// refreshSubConns creates/removes SubConns with backendAddrs, and refreshes
// balancer state and picker.
//
// Caller must hold lb.mu.
func (lb *lbBalancer) refreshSubConns(backendAddrs []resolver.Address, fromGRPCLBServer bool) {
	defer func() {
		// Regenerate and update picker after refreshing subconns because with
		// cache, even if SubConn was newed/removed, there might be no state
		// changes (the subconn will be kept in cache, not actually
		// newed/removed).
		lb.updateStateAndPicker(true, true)
	}()

	lb.inFallback = !fromGRPCLBServer

	opts := balancer.NewSubConnOptions{}
	if fromGRPCLBServer {
		opts.CredsBundle = lb.grpclbBackendCreds
	}
	lb.backendAddrs = nil
	if lb.usePickFirst {
		var sc balancer.SubConn
		for _, sc = range lb.subConns {
			break
		}
		if sc != nil {
			sc.UpdateAddresses(backendAddrs)
			sc.Connect()
			return
		}
		// This bypasses the cc wrapper with SubConn cache.
		sc, err := lb.cc.cc.NewSubConn(backendAddrs, opts)
		if err != nil {
			grpclog.Warningf("grpclb: failed to create new SubConn: %v", err)
			return
		}
		sc.Connect()
		lb.subConns[backendAddrs[0]] = sc
		lb.scStates[sc] = connectivity.Idle
		return
	}
	// addrsSet is the set converted from backendAddrs, it's used to quick
	// lookup for an address.
	addrsSet := make(map[resolver.Address]struct{})
	// Create new SubConns.
	for _, addr := range backendAddrs {
		addrWithoutMD := addr
		addrWithoutMD.Metadata = nil
		addrsSet[addrWithoutMD] = struct{}{}
		lb.backendAddrs = append(lb.backendAddrs, addrWithoutMD)
		if _, ok := lb.subConns[addrWithoutMD]; !ok {
			// Use addrWithMD to create the SubConn.
			sc, err := lb.cc.NewSubConn([]resolver.Address{addr}, opts)
			if err != nil {
				grpclog.Warningf("grpclb: failed to create new SubConn: %v", err)
				continue
			}
			lb.subConns[addrWithoutMD] = sc // Use the addr without MD as key for the map.
			if _, ok := lb.scStates[sc]; !ok {
				// Only set state of new sc to IDLE. The state could already be
				// READY for cached SubConns.
				lb.scStates[sc] = connectivity.Idle
			}
			sc.Connect()
		}
	}
	for a, sc := range lb.subConns {
		// a was removed by resolver.
		if _, ok := addrsSet[a]; !ok {
			lb.cc.RemoveSubConn(sc)
			delete(lb.subConns, a)
			// Keep the state of this sc in b.scStates until sc's state becomes Shutdown.
			// The entry will be deleted in HandleSubConnStateChange.
		}
	}
}
###### 7 ######
URL: https://github.com/livepeer/go-livepeer/pull/657#discussion_r244421875

Review:
Update comment to reflect the update from a single sessionID to a list of sessionIDs?
Old Version:
// RedeemWinningTicket redeems all winning tickets with the broker
// for a session ID
func (r *recipient) RedeemWinningTickets(sessionIDs []string) error {
	tickets, sigs, recipientRands, err := r.store.LoadWinningTickets(sessionIDs)
	if err != nil {
		return err
	}
	for i := 0; i < len(tickets); i++ {
		if err := r.redeemWinningTicket(tickets[i], sigs[i], recipientRands[i]); err != nil {
			return err
		}
	}
	return nil
}
New Version:
// RedeemWinningTicket redeems all winning tickets with the broker
// for a all sessionIDs
func (r *recipient) RedeemWinningTickets(sessionIDs []string) error {
	tickets, sigs, recipientRands, err := r.store.LoadWinningTickets(sessionIDs)
	if err != nil {
		return err
	}
	for i := 0; i < len(tickets); i++ {
		if err := r.redeemWinningTicket(tickets[i], sigs[i], recipientRands[i]); err != nil {
			return err
		}
	}
	return nil
}
###### 8 ######
URL: https://github.com/i-love-flamingo/flamingo/commit/d678d9318a2e
Review: fix godoc
Old version:
// Identify an incoming request
func (m *Identifier) Authenticate(ctx context.Context, request *web.Request) web.Result {
        if m.authenticateMethod != nil {
                return m.authenticateMethod(m, ctx, request)
        }
        return nil
}
New version:
// Authenticate an incoming request
func (m *Identifier) Authenticate(ctx context.Context, request *web.Request) web.Result {
        if m.authenticateMethod != nil {
                return m.authenticateMethod(m, ctx, request)
        }
        return nil
}
###### 9 ######
URL: https://github.com/mailgun/mailgun-go/commit/ac471bcc1eba
Review: Fix godoc
Old version:
// GetStoredMessage retrieves information about a received e-mail message.
// This provides visibility into, e.g., replies to a message sent to a mailing list.
func (mg *MailgunImpl) GetStoredMessageRaw(id string) (StoredMessageRaw, error) {
        url := generateStoredMessageUrl(mg, messagesEndpoint, id)
        r := simplehttp.NewHTTPRequest(url)
        r.SetBasicAuth(basicAuthUser, mg.ApiKey())
        r.AddHeader("Accept", "message/rfc2822")

        var response StoredMessageRaw
        err := getResponseFromJSON(r, &response)
        return response, err

}
New version:
// GetStoredMessageRaw retrieves the raw MIME body of a received e-mail message.
// Compared to GetStoredMessage, it gives access to the unparsed MIME body, and
// thus delegates to the caller the required parsing.
func (mg *MailgunImpl) GetStoredMessageRaw(id string) (StoredMessageRaw, error) {
        url := generateStoredMessageUrl(mg, messagesEndpoint, id)
        r := simplehttp.NewHTTPRequest(url)
        r.SetBasicAuth(basicAuthUser, mg.ApiKey())
        r.AddHeader("Accept", "message/rfc2822")

        var response StoredMessageRaw
        err := getResponseFromJSON(r, &response)
        return response, err

}
###### 10 ######
URL: https://github.com/gopcua/opcua/commit/22307d80b73e
Review: fix comment
Old version:
// New creates a new NodeMonitor
func NewNodeMonitor(client *opcua.Client) (*NodeMonitor, error) {
        m := &NodeMonitor{
                client:           client,
                nextClientHandle: 100,
        }

        return m, nil
}
New version:
// NewNodeMonitor creates a new NodeMonitor
func NewNodeMonitor(client *opcua.Client) (*NodeMonitor, error) {
        m := &NodeMonitor{
                client:           client,
                nextClientHandle: 100,
        }

        return m, nil
}
###### 11 ######
URL: https://github.com/larien/aprenda-go-com-testes/commit/60e25c9acc92
Review: fix comment
Old version:
// GetLeague currently doesn't work, but it should return the player league
func (i *InMemoryPlayerStore) GetLeague() []Player {
        var league []Player
        for name, wins := range i.store {
                league = append(league, Player{name, wins})
        }
        return league
}
New version:
// GetLeague returns a collection of Players
func (i *InMemoryPlayerStore) GetLeague() []Player {
        var league []Player
        for name, wins := range i.store {
                league = append(league, Player{name, wins})
        }
        return league
}
###### 12 ######
URL: https://github.com/opencost/opencost/commit/23ee52889394
Review: Comment fix
Old version:
// Sets the bool value atomically
func (ai *AtomicInt32) Set(value int32) {
        atomic.StoreInt32((*int32)(ai), value)
}
New version:
// Sets the int32 value atomically
func (ai *AtomicInt32) Set(value int32) {
        atomic.StoreInt32((*int32)(ai), value)
}
###### 13 ######
URL: https://github.com/smallstep/cli/commit/6ad024e90e54
Review: Fix comment
Old version:
// CreateAttestor creates an attestor that will use `step-kms-plugin` with the
// given kms and uri.
func CreateAttestor(kms, name string) (Attestor, error) {
        return newKMSSigner(kms, name)
}
New version:
// CreateAttestor creates an attestor that will use `step-kms-plugin` with the
// given kms and name.
func CreateAttestor(kms, name string) (Attestor, error) {
        return newKMSSigner(kms, name)
}
###### 14 ######
URL: https://github.com/FeatureBaseDB/featurebase/commit/4013cccb303a
Review: fix comment
Old version:
// Close implements the Closer interface.
func (g *memberSet) Close() error {
        leaveErr := g.memberlist.Leave(5 * time.Second)
        shutdownErr := g.memberlist.Shutdown()
        if leaveErr != nil || shutdownErr != nil {
                return fmt.Errorf("leaving: '%v', shutting down: '%v'", leaveErr, shutdownErr)
        }
        return nil
}
New version:
// Close attempts to gracefully leaves the cluster, and finally calls shutdown
// after (at most) a timeout period.
func (g *memberSet) Close() error {
        leaveErr := g.memberlist.Leave(5 * time.Second)
        shutdownErr := g.memberlist.Shutdown()
        if leaveErr != nil || shutdownErr != nil {
                return fmt.Errorf("leaving: '%v', shutting down: '%v'", leaveErr, shutdownErr)
        }
        return nil
}
###### 15 ######
URL: https://github.com/OffchainLabs/arbitrum-classic/commit/7b1112509fb4
Review: Fix comment
Old version:
// CallMessage takes a request from a client to process in a temporary context
// and return the result
func (m *RPCServer) PendingCall(
        _ *http.Request,
        args *validatorserver.CallMessageArgs,
        reply *validatorserver.CallMessageReply,
) error {
        ret, err := m.Server.PendingCall(context.Background(), args)
        if err != nil || ret == nil {
                return err
        }
        reply.RawVal = ret.RawVal
        return nil
}
New version:
// PendingCall takes a request from a client to process in a temporary context
// and return the result
func (m *RPCServer) PendingCall(
        _ *http.Request,
        args *validatorserver.CallMessageArgs,
        reply *validatorserver.CallMessageReply,
) error {
        ret, err := m.Server.PendingCall(context.Background(), args)
        if err != nil || ret == nil {
                return err
        }
        reply.RawVal = ret.RawVal
        return nil
}
###### 16 ######
URL: https://github.com/helm/helm/commit/c55defe15b3f
Review: Fix comment
Old version:
// GetRepo returns the repository with the given URL
func (m *manager) GetRepo(repoName string) (repo.IRepo, error) {
        repoURL, err := m.service.GetRepoURLByName(repoName)
        if err != nil {
                return nil, err
        }

        return m.service.GetRepoByURL(repoURL)
}
New version:
// GetRepo returns the repository with the given name
func (m *manager) GetRepo(repoName string) (repo.IRepo, error) {
        repoURL, err := m.service.GetRepoURLByName(repoName)
        if err != nil {
                return nil, err
        }

        return m.service.GetRepoByURL(repoURL)
}
###### 17 ######
URL: https://github.com/weaveworks/scope/commit/4444a405e0f8
Review: Fix comment
Old version:
// Hostname returns the hostname of this host.
func Get() string {
        if hostname := os.Getenv("SCOPE_HOSTNAME"); hostname != "" {
                return hostname
        }
        hostname, err := os.Hostname()
        if err != nil {
                return "(unknown)"
        }
        return hostname
}
New version:
// Get returns the hostname of this host.
func Get() string {
        if hostname := os.Getenv("SCOPE_HOSTNAME"); hostname != "" {
                return hostname
        }
        hostname, err := os.Hostname()
        if err != nil {
                return "(unknown)"
        }
        return hostname
}
###### 18 ######
URL: https://github.com/FairwindsOps/nova/commit/74a025afef68
Review: fix comment
Old version:
// Send dispatches a message to file
func (output Output) ToFile(filename string) error {
        data, err := json.Marshal(output)
        if err != nil {
                klog.Errorf("Error marshaling json: %v", err)
                return err
        }

        err = ioutil.WriteFile(filename, data, 0644)
        if err != nil {
                klog.Errorf("Error writing to file %s: %v", filename, err)
        }
        return nil
}
New version:
// ToFile dispatches a message to file
func (output Output) ToFile(filename string) error {
        data, err := json.Marshal(output)
        if err != nil {
                klog.Errorf("Error marshaling json: %v", err)
                return err
        }

        err = ioutil.WriteFile(filename, data, 0644)
        if err != nil {
                klog.Errorf("Error writing to file %s: %v", filename, err)
        }
        return nil
}
###### 19 ######
URL: https://github.com/target/goalert/commit/4a2918f41ee0
Review: fix comment
Old version:
// NewAlertLog will generate an alert log with the provided status.
func (d *datagen) NewAlertLogs(alert alert.Alert) {

        // Add 'created' event log
        d.AlertLogs = append(d.AlertLogs, AlertLog{
                AlertID:   alert.ID,
                Timestamp: alert.CreatedAt,
                Event:     "created",
                Message:   "",
        })

        // Add 'closed' event log
        if alert.Status == "closed" {
                d.AlertLogs = append(d.AlertLogs, AlertLog{
                        AlertID:   alert.ID,
                        Timestamp: gofakeit.DateRange(alert.CreatedAt, alert.CreatedAt.Add(30*time.Minute)),
                        Event:     "closed",
                        Message:   "",
                })
        }
}
New version:
// NewAlertLog will generate an alert log for the provided alert.
func (d *datagen) NewAlertLogs(alert alert.Alert) {

        // Add 'created' event log
        d.AlertLogs = append(d.AlertLogs, AlertLog{
                AlertID:   alert.ID,
                Timestamp: alert.CreatedAt,
                Event:     "created",
                Message:   "",
        })

        // Add 'closed' event log
        if alert.Status == "closed" {
                d.AlertLogs = append(d.AlertLogs, AlertLog{
                        AlertID:   alert.ID,
                        Timestamp: gofakeit.DateRange(alert.CreatedAt, alert.CreatedAt.Add(30*time.Minute)),
                        Event:     "closed",
                        Message:   "",
                })
        }
}
###### 20 ######
URL: https://github.com/go-mysql-org/go-mysql/commit/cc2bb58b8898
Review: Fixing comment
Old version:
// SetTableCache sets table cache value for the given key
func (c *Canal) SetTableCache(db []byte, table []byte, schema *schema.Table) {
        key := fmt.Sprintf("%s.%s", db, table)
        c.tableLock.Lock()
        c.tables[key] = schema
        if c.cfg.DiscardNoMetaRowEvent {
                // if get table info success, delete this key from errorTablesGetTime
                delete(c.errorTablesGetTime, key)
        }
        c.tableLock.Unlock()
}
New version:
// SetTableCache sets table cache value for the given table
func (c *Canal) SetTableCache(db []byte, table []byte, schema *schema.Table) {
        key := fmt.Sprintf("%s.%s", db, table)
        c.tableLock.Lock()
        c.tables[key] = schema
        if c.cfg.DiscardNoMetaRowEvent {
                // if get table info success, delete this key from errorTablesGetTime
                delete(c.errorTablesGetTime, key)
        }
        c.tableLock.Unlock()
}
###### 21 ######
URL: https://github.com/thanos-io/thanos/commit/b3d81ec8254b
Review: Improve comments, fix nits
Old version:
// Start runs the Prometheus instance until the context is canceled.
func (p *Prometheus) Start() error {
        p.running = true
        if err := p.db.Close(); err != nil {
                return err
        }

        p.cmd = exec.Command(
                "prometheus",
                "--storage.tsdb.path="+p.dir,
                "--web.listen-address="+p.addr,
                "--config.file="+filepath.Join(p.dir, "prometheus.yml"),
        )
        go func() {
                if b, err := p.cmd.CombinedOutput(); err != nil {
                        fmt.Fprintln(os.Stderr, "running Prometheus failed", err)
                        fmt.Fprintln(os.Stderr, string(b))
                }
        }()
        time.Sleep(time.Second)

        return nil
}
New version:
// Start running the Prometheus instance and return.
func (p *Prometheus) Start() error {
        p.running = true
        if err := p.db.Close(); err != nil {
                return err
        }

        p.cmd = exec.Command(
                "prometheus",
                "--storage.tsdb.path="+p.dir,
                "--web.listen-address="+p.addr,
                "--config.file="+filepath.Join(p.dir, "prometheus.yml"),
        )
        go func() {
                if b, err := p.cmd.CombinedOutput(); err != nil {
                        fmt.Fprintln(os.Stderr, "running Prometheus failed", err)
                        fmt.Fprintln(os.Stderr, string(b))
                }
        }()
        time.Sleep(time.Second)

        return nil
}
###### 22 ######
URL: https://github.com/okteto/okteto/commit/a4897660a3e5
Review: Documentation grammar fixes (#3216)

* Multiple documentation grammmar fixes

Signed-off-by: Abhisman Sarkar <abhisman.sarkar@gmail.com>

* Add suggested doc changes

Signed-off-by: Abhisman Sarkar <abhisman.sarkar@gmail.com>

Signed-off-by: Abhisman Sarkar <abhisman.sarkar@gmail.com>
Old version:
// MinimumNArgsAccepted returns an error if there are more than N args.
func MinimumNArgsAccepted(n int, url string) cobra.PositionalArgs {
        return func(cmd *cobra.Command, args []string) error {
                var hint string
                if url != "" {
                        hint = fmt.Sprintf("Visit %s for more information.", url)
                }
                if len(args) < n {
                        return oktetoErrors.UserError{
                                E:    fmt.Errorf("%q requires at least %d arg(s), but only received %d", cmd.CommandPath(), n, len(args)),
                                Hint: hint,
                        }
                }
                return nil
        }
}
New version:
// MinimumNArgsAccepted returns an error if there are less than N args.
func MinimumNArgsAccepted(n int, url string) cobra.PositionalArgs {
        return func(cmd *cobra.Command, args []string) error {
                var hint string
                if url != "" {
                        hint = fmt.Sprintf("Visit %s for more information.", url)
                }
                if len(args) < n {
                        return oktetoErrors.UserError{
                                E:    fmt.Errorf("%q requires at least %d arg(s), but only received %d", cmd.CommandPath(), n, len(args)),
                                Hint: hint,
                        }
                }
                return nil
        }
}
###### 23 ######
URL: https://github.com/gonum/plot/commit/12787dd210cb
Review: Fix comments.
Old version:
// LocScale is a scale function for a log-scale axis.
func LogScale(min, max, x float64) float64 {
        logMin := log(min)
        return (log(x) - logMin) / (log(max) - logMin)
}
New version:
// LocScale can be used as the value of an Axis.Scale function to
// set the axis to a log scale.
func LogScale(min, max, x float64) float64 {
        logMin := log(min)
        return (log(x) - logMin) / (log(max) - logMin)
}
###### 24 ######
URL: https://github.com/nutsdb/nutsdb/commit/f413465f063e
Review: Fix error comment
Old version:
// ZPeekMax returns up to count members with the highest scores in the sorted set stored at key.
func (tx *Tx) ZPeekMax(bucket string) (*zset.SortedSetNode, error) {
        if err := tx.checkTxIsClosed(); err != nil {
                return nil, err
        }

        if _, ok := tx.db.SortedSetIdx[bucket]; !ok {
                return nil, ErrBucket
        }

        return tx.db.SortedSetIdx[bucket].PeekMax(), nil
}
New version:
// ZPeekMax returns up to count members with the highest scores in the sorted set stored at bucket.
func (tx *Tx) ZPeekMax(bucket string) (*zset.SortedSetNode, error) {
        if err := tx.checkTxIsClosed(); err != nil {
                return nil, err
        }

        if _, ok := tx.db.SortedSetIdx[bucket]; !ok {
                return nil, ErrBucket
        }

        return tx.db.SortedSetIdx[bucket].PeekMax(), nil
}
###### 25 ######
URL: https://github.com/GoogleContainerTools/kpt/commit/e41decf62d90
Review: git CommitHelper: a few small fixes (#3012)

Rationalizing some of the codepaths and adding some comments per my
understanding.
Old version:
// Returns index of the entry if found (by name); nil if not found
func findEntry(tree *object.Tree, name string) *object.TreeEntry {
        for i := range tree.Entries {
                e := &tree.Entries[i]
                if e.Name == name {
                        return e
                }
        }
        return nil
}
New version:
// Returns a pointer to the entry if found (by name); nil if not found
func findEntry(tree *object.Tree, name string) *object.TreeEntry {
        for i := range tree.Entries {
                e := &tree.Entries[i]
                if e.Name == name {
                        return e
                }
        }
        return nil
}