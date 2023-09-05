This is the full benchmark set for Python of 25 cases

###### 1 ######
URL: https://github.com/astropy/astropy/pull/3067#discussion_r19736856

Review: Just update this docstring and this should be fine.  Something more along the lines of what it actually tests.  For example
```
"""
Returns `True` if ``HDUList.index_of(item)`` succeeds.
"""
```

Old Version:
def __contains__(self, item):
    """
    Used by the 'in' operator
    """
    try:
        self.index_of(item)
        return True
    except KeyError:
        return False

New Version:
def __contains__(self, item):
    """
    Returns `True` if HDUList.index_of(item) succeeds.
    """
    try:
        self.index_of(item)
        return True
    except KeyError:
        return False

###### 2 ######
URL: https://github.com/pretix/pretix/pull/334#discussion_r89618197

Review: Please update the docstring here, aswell.

Old Version:
def is_allowed(self, request: HttpRequest) -> bool:
    """
    You can use this method to disable this payment provider for certain groups
    of users, products or other criteria. If this method returns ``False``, the
    user will not be able to select this payment method. This will only be called
    during checkout, not on retrying.

    The default implementation always returns ``True``.
    """
    return self._is_still_available()

New Version:
def is_allowed(self, request: HttpRequest) -> bool:
    """
    You can use this method to disable this payment provider for certain groups
    of users, products or other criteria. If this method returns ``False``, the
    user will not be able to select this payment method. This will only be called
    during checkout, not on retrying.

    The default implementation checks for the _availability_date setting to be either unset or in the future.
    """
    return self._is_still_available()

###### 3 ######
URL: https://github.com/matplotlib/matplotlib/pull/5942#discussion_r51233494

Review: Should the docstring be updated with this change

Old Version:
def option_scale_image(self):
    """
    agg backend support arbitrary scaling of image.
    """
    return True

New Version:
def option_scale_image(self):
    """
    agg backend doesn't support arbitrary scaling of image.
    """
    return False

###### 4 ######
URL: https://github.com/TheAlgorithms/Python/pull/3115#discussion_r505426662

Review: 
Does it check whether n is pandigital or palindromic? Please update either the docstring or function name accordingly.

Old Version:
def is_9_palindromic(n: int) -> bool:
    """
    Checks whether n is a 9-digit 1 to 9 pandigital number.
    >>> is_9_palindromic(12345)
    False
    >>> is_9_palindromic(156284973)
    True
    >>> is_9_palindromic(1562849733)
    False
    """
    s = str(n)
    return len(s) == 9 and set(s) == set("123456789")

New Version:
def is_9_pandigital(n: int) -> bool:
    """
    Checks whether n is a 9-digit 1 to 9 pandigital number.
    >>> is_9_pandigital(12345)
    False
    >>> is_9_pandigital(156284973)
    True
    >>> is_9_pandigital(1562849733)
    False
    """
    s = str(n)
    return len(s) == 9 and set(s) == set("123456789")

###### 5 ######
URL: https://github.com/SwissDataScienceCenter/renku-python/pull/3047#discussion_r931177369

Review: This won't change and makes sense to be a tuple. I've updated the docstring.

Old Version:
@staticmethod
@abc.abstractmethod
def get_credentials_names() -> Tuple[str, ...]:
    """Return list of the required credentials for a provider."""
    raise NotImplementedError

New Version:
@staticmethod
@abc.abstractmethod
def get_credentials_names() -> Tuple[str, ...]:
    """Return a tuple of the required credentials for a provider."""
    raise NotImplementedError

###### 6 ######
URL: https://github.com/metoppv/improver/pull/875#discussion_r294704128

Review:
Please update docstring.
Old Version:
    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_realizations_predictor_keyerror(self):
        """
        Test that the minimisation has resulted in a successful convergence,
        and that the object returned is an OptimizeResult object, when the
        ensemble realizations are the predictor.
        """
        predictor_of_mean_flag = "realizations"
        distribution = "foo"

        plugin = Plugin()
        msg = "Distribution requested"
        with self.assertRaisesRegex(KeyError, msg):
            plugin.crps_minimiser_wrapper(
                self.initial_guess_for_realization,
                self.forecast_predictor_realizations, self.truth,
                self.forecast_variance, predictor_of_mean_flag, distribution)


New Version:
    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_realizations_predictor_keyerror(self):
        """
        Test that an exception is raised when the distribution requested is
        not an available option when the predictor_of_mean_flag is the
        ensemble realizations.
        """
        predictor_of_mean_flag = "realizations"
        distribution = "foo"

        plugin = Plugin()
        msg = "Distribution requested"
        with self.assertRaisesRegex(KeyError, msg):
            plugin.crps_minimiser_wrapper(
                self.initial_guess_for_realization,
                self.forecast_predictor_realizations, self.truth,
                self.forecast_variance, predictor_of_mean_flag, distribution)

###### 7 ######
URL: https://github.com/primeqa/primeqa/pull/321#discussion_r1082098230

Review:
        \"\"\"\r\n        Our approach to pruning hallucinated questions uses an entity lookup to check that \r\n        a generated entity or a part of it is present in the hybrid context. If the entity is not present, \r\n        the question and the entity are considered to be hallucinated.\r\n\r\n        Args:\r\n            qdicts (list): A list of dicts, each dict contains, an answer, context and (num_instances) generated questions.\r\n        Returns:\r\n            Updated qdicts without any hallucinated questions. Each dict will have only one question now.\r\n        \"\"\"\r\nI have pushed the changes with this docstring.\r\n
Old Version:
    def prune_hallucinations(self, qdicts, num_instances=5, hallucination_prop=0.25):
        """
            Prunes hallucinated questions using an entity lookup approach.
        Args:
            qdicts (list): A list of dicts, each dict contains, an answer, context and (num_instances) generated questions.
        Returns:
            Updated qdicts without any hallucinated questions. Each dict will have only one question now.
        """
        new_qdicts = []
        reserve_hallucinated_qdict = []
        reserve_non_hallucinated_qdict = []
        question_words = set(['what', 'when', 'where', 'which', 'how', 'who', 'whose', 'whom'])
        for qdict in qdicts:
            answer = qdict['answer']
            context = qdict.pop('context')
            questions = qdict.pop('questions')
            if not reserve_hallucinated_qdict:
                qdict['question'] = questions[0]
                reserve_hallucinated_qdict.append(qdict)
            context_tokens = set([tok.text.lower() for tok in self.path_sampler.nlp_model.process(\
                                context, processors='tokenize').sentences[0].tokens])
            flag = False
            same_questions = set([])
            for question in questions:
                if not question.strip() or question in same_questions: continue
                same_questions.add(question)
                doc = self.path_sampler.nlp_model(question)
                hallucinated = set([])
                for entity in doc.sentences[0].entities:
                    entity = entity.text.lower()
                    entity_tokens = set([etok.text.strip().lower() for etok in self.path_sampler.nlp_model.process(\
                                        entity, processors='tokenize').sentences[0].tokens if etok.text.strip().lower() != "'s"])
                    has_qs_words = question_words.intersection(entity_tokens)
                    if has_qs_words: continue

                    hcount = 0
                    for e_tok in entity_tokens:
                        if e_tok not in context_tokens:
                            hcount += 1

                    if hcount == 0: continue

                    hall_quant = round(len(entity_tokens) * hallucination_prop) #no. of hall. words tolerable
                    if hcount > hall_quant or len(entity_tokens) == hcount:
                        hallucinated.add(entity)

                if not hallucinated:
                    qdict_copy = copy.deepcopy(qdict)
                    qdict_copy['question'] = question
                    if flag:
                        reserve_non_hallucinated_qdict.append(qdict_copy)
                    else:
                        new_qdicts.append(qdict_copy)
                    flag = True

        if not new_qdicts:
            return reserve_hallucinated_qdict
        elif len(new_qdicts) < num_instances:
            diff = num_instances - len(new_qdicts)
            np.random.shuffle(reserve_non_hallucinated_qdict) 
            return new_qdicts + reserve_non_hallucinated_qdict[:diff]
        else: 
            return new_qdicts

New Version:
    def prune_hallucinations(self, qdicts, num_instances=5, hallucination_prop=0.25):
        """
        Our approach to pruning hallucinated questions uses an entity lookup to check that 
        a generated entity or a part of it is present in the hybrid context. If the entity is not present, 
        the question and the entity are considered to be hallucinated.

        Args:
            qdicts (list): A list of dicts, each dict contains, an answer, context and (num_instances) generated questions.
        Returns:
            Updated qdicts without any hallucinated questions. Each dict will have only one question now.
        """
        new_qdicts = []
        reserve_hallucinated_qdict = []
        reserve_non_hallucinated_qdict = []
        question_words = set(['what', 'when', 'where', 'which', 'how', 'who', 'whose', 'whom'])
        for qdict in qdicts:
            answer = qdict['answer']
            context = qdict.pop('context')
            questions = qdict.pop('questions')
            if not reserve_hallucinated_qdict:
                qdict['question'] = questions[0]
                reserve_hallucinated_qdict.append(qdict)
            context_tokens = set([tok.text.lower() for tok in self.path_sampler.nlp_model.process(\
                                context, processors='tokenize').sentences[0].tokens])
            flag = False
            same_questions = set([])
            for question in questions:
                if not question.strip() or question in same_questions: continue
                same_questions.add(question)
                doc = self.path_sampler.nlp_model(question)
                hallucinated = set([])
                for entity in doc.sentences[0].entities:
                    entity = entity.text.lower()
                    entity_tokens = set([etok.text.strip().lower() for etok in self.path_sampler.nlp_model.process(\
                                        entity, processors='tokenize').sentences[0].tokens if etok.text.strip().lower() != "'s"])
                    has_qs_words = question_words.intersection(entity_tokens)
                    if has_qs_words: continue

                    hcount = 0
                    for e_tok in entity_tokens:
                        if e_tok not in context_tokens:
                            hcount += 1

                    if hcount == 0: continue

                    hall_quant = round(len(entity_tokens) * hallucination_prop) #no. of hall. words tolerable
                    if hcount > hall_quant or len(entity_tokens) == hcount:
                        hallucinated.add(entity)

                if not hallucinated:
                    qdict_copy = copy.deepcopy(qdict)
                    qdict_copy['question'] = question
                    if flag:
                        reserve_non_hallucinated_qdict.append(qdict_copy)
                    else:
                        new_qdicts.append(qdict_copy)
                    flag = True

        if not new_qdicts:
            #don't prune if all the generated questions are hallucinated 
            return reserve_hallucinated_qdict
        elif len(new_qdicts) < num_instances:
            #if number of questions are less than `num_instances` after pruning
            #use pruned questions to return number of questions equal to `num_instances`
            diff = num_instances - len(new_qdicts)
            np.random.shuffle(reserve_non_hallucinated_qdict) 
            return new_qdicts + reserve_non_hallucinated_qdict[:diff]
        else: 
            return new_qdicts

###### 8 ######
URL: https://github.com/Qiskit-Partners/qiskit-ionq/pull/52#discussion_r604487898

Review:
these are just extraneous docstring fixes, as I realized they were incorrect copy-pastes before

Old Version:
def test_output_map__with_multiple_measurements_to_different_clbits(simulator_backend):
    """Test a full circuit

    Args:
        simulator_backend (IonQSimulatorBackend): A simulator backend fixture.
    """
    qc = QuantumCircuit(2, 2, name="test_name")
    qc.measure(0, 0)
    qc.measure(0, 1)
    ionq_json = qiskit_to_ionq(
        qc,
        simulator_backend.name(),
        passed_args={"shots": 200},
    )
    actual = json.loads(ionq_json)
    actual_maps = actual.pop("registers") or {}
    actual_output_map = actual_maps.pop("meas_mapped")

    assert actual_output_map == [0, 0]

New Version:
def test_output_map__with_multiple_measurements_to_different_clbits(simulator_backend):
    """Test output mapping handles multiple measurements from the same qubit to different clbits correctly

    Args:
        simulator_backend (IonQSimulatorBackend): A simulator backend fixture.
    """
    qc = QuantumCircuit(2, 2, name="test_name")
    qc.measure(0, 0)
    qc.measure(0, 1)
    ionq_json = qiskit_to_ionq(
        qc,
        simulator_backend.name(),
        passed_args={"shots": 200},
    )
    actual = json.loads(ionq_json)
    actual_maps = actual.pop("registers") or {}
    actual_output_map = actual_maps.pop("meas_mapped")

    assert actual_output_map == [0, 0]


###### 9 ######
URL: https://github.com/pytorch/ignite/pull/1714/files/f06592e3c6ebf3f466f0b445dc257370cfbd3d39#r585571973

Review:
Suggested change
    Factory function for creating an evaluator for supervised models.
    Factory function for supervised evaluation.

To be consistent with supervised_training_*

Old Version:
def supervised_evaluation_step(
    model: torch.nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred: (y_pred, y),
) -> Callable:
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model: the model to train.
        device: device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform: function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.

    Returns:
        an evaluator engine with supervised inference function.

    Note:
        `engine.state.output` for this engine is defind by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    .. warning::

        The internal use of `device` has changed.
        `device` will now *only* be used to move the input data to the correct device.
        The `model` should be moved by the user before creating an optimizer.

        For more information see:

        - `PyTorch Documentation <https://pytorch.org/docs/stable/optim.html#constructing-it>`_

        - `PyTorch's Explanation <https://github.com/pytorch/pytorch/issues/7844#issuecomment-503713840>`_

    .. versionchanged:: 0.5.0
    """

    def evaluate_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            return output_transform(x, y, y_pred)

    return evaluate_step

New Version:
def supervised_evaluation_step(
    model: torch.nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred: (y_pred, y),
) -> Callable:
    """
    Factory function for supervised evaluation.

    Args:
        model: the model to train.
        device: device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform: function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.

    Returns:
        Inference function.

    Note:
        `engine.state.output` for this engine is defind by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    .. warning::

        The internal use of `device` has changed.
        `device` will now *only* be used to move the input data to the correct device.
        The `model` should be moved by the user before creating an optimizer.

    .. versionadded:: 0.5.0
    """

    def evaluate_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            return output_transform(x, y, y_pred)

    return evaluate_step

###### 10 ######
URL: https://github.com/matrix-org/synapse/pull/3574#discussion_r208528659

Review:
grr. can you fix the docstring while you are there?

Old Version:
    def get_user_count_txn(self, txn):
        """Get a total number of registerd users in the users list.
        Args:
            txn : Transaction object
        Returns:
            defer.Deferred: resolves to int
        """
        sql_count = "SELECT COUNT(*) FROM users WHERE is_guest = 0;"
        txn.execute(sql_count)
        count = txn.fetchone()[0]
        defer.returnValue(count)

New Version:
    def get_user_count_txn(self, txn):
        """Get a total number of registered users in the users list.

        Args:
            txn : Transaction object
        Returns:
            int : number of users
        """
        sql_count = "SELECT COUNT(*) FROM users WHERE is_guest = 0;"
        txn.execute(sql_count)
        return txn.fetchone()[0]

###### 11 ######
URL: https://github.com/red-hat-storage/ocs-ci/pull/40#discussion_r287815310

Review:
This function doesn't return just random 13 chars but whatever size you define, so please change the docstring accordingly. I won't block this PR and merge it now, but plese fix in some next PR @vavuthu \r\n```python\r\nargs:\r\n    size (int)\r\n```\r\n\r\nI see you don't have any other approve yet, so maybe you can fix it in this PR and we can merge tomorrow.\r\n
Old Version:
def get_random_str(size=13):
    """
    generates the random string of 13 characters

    Returns:
         str : string of random 13 characters

    """
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))

New Version:
def get_random_str(size=13):
    """
    generates the random string of given size

    Args:
        size (int): number of random characters to generate

    Returns:
         str : string of random characters of given size

    """
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))

###### 12 ######
URL: https://github.com/pynucastro/pynucastro/pull/330#discussion_r918935363

Review:
can you make the docstring more descriptive? What is the purpose of this function now?

Old Version:
    def derived_forward(self):
        """
        We exclude the weak and tabular rates from the .foward() library.
        """

        collect_rates = []
        onlyfwd = self.forward()

        for r in onlyfwd.get_rates():

            try:
                DerivedRate(r, use_pf=True, use_A_nuc=True)
            except ValueError:
                continue
            else:
                collect_rates.append(r)

        list1 = Library(rates=collect_rates)
        return list1

New Version:
    def derived_forward(self):
        """
        In this library, We exclude the weak and tabular rates from the .foward() library which includes all
        the ReacLib forward reactions.

        In a future PR, we will classify forward reactions as exothermic (Q>0), and reverse by endothermic (Q<0).
        However, ReacLib does not follow this path. If a reaction is measured experimentally (independent of Q),
        they use detailed balance to get the opposite direction. Eventually, I want to classify forward and reverse
        by positive Q and negative Q; however, for testing purposes, making this classification may eventually lead to
        computing the detailed balance twice.

        The idea of derived_forward is to eliminate the reverse and weak, and see if our job gives the same Reaclib
        predictions, checking the NSE convergence with the pf functions. In the future, I want to move this function
        in a unit test.
        """

        collect_rates = []
        onlyfwd = self.forward()

        for r in onlyfwd.get_rates():

            try:
                DerivedRate(r, use_pf=True, use_A_nuc=True)
            except ValueError:
                continue
            else:
                collect_rates.append(r)

        list1 = Library(rates=collect_rates)
        return list1

###### 13 ######
URL: https://github.com/movingpandas/movingpandas/pull/100#discussion_r569232659

Review:
The docstring will need to be updated to reflect changes. There are probably other docstrings that are related to this as well. 
Old Version:
def _measure_distance(point1, point2, spherical=False):
    """
    Convenience function that returns either euclidean or spherical distance between two points
    """
    if spherical:
        return measure_distance_spherical(point1, point2)
    else:
        return measure_distance_euclidean(point1, point2)

New Version:
def _measure_distance(point1, point2, spherical=False):
    """
    Convenience function that returns either euclidean or geodesic distance between two points
    """
    if spherical:
        return measure_distance_geodesic(point1, point2)
    else:
        return measure_distance_euclidean(point1, point2)
###### 14 ######
URL: https://github.com/ReactionMechanismGenerator/RMG-Py/pull/1260/files#r166817460

Review:
Although not part of this PR, could you fix the docstring of isOrder()? It doesn't check only for single bonds
Old Version:
    def isOrder(self, otherOrder):
        """
        Return ``True`` if the bond represents a single bond or ``False`` if
        not. This compares floats that takes into account floating point error
        
        NOTE: we can replace the absolute value relation with math.isclose when
        we swtich to python 3.5+
        """
        return abs(self.order - otherOrder) <= 1e-4
New Version:
    def isOrder(self, otherOrder):
        """
        Return ``True`` if the bond is of order otherOrder or ``False`` if
        not. This compares floats that takes into account floating point error
        
        NOTE: we can replace the absolute value relation with math.isclose when
        we swtich to python 3.5+
        """
        return abs(self.order - otherOrder) <= 1e-4
###### 15 ######
URL: https://github.com/aiidateam/aiida-core/pull/4470#discussion_r508120179

Review:
Up to you - in any case, the docstring should be consistent and if you prefer not to fix it here, we should open an issue

Old Version:
    def pop(self, **kwargs):  # pylint: disable=arguments-differ
        """Remove and return item at index (default last)."""
        data = self.get_list()
        data.pop(**kwargs)
        if not self._using_list_reference():
            self.set_list(data)
New Version:
    def pop(self, **kwargs):  # pylint: disable=arguments-differ
        """Remove and return item at index (default last)."""
        data = self.get_list()
        item = data.pop(**kwargs)
        if not self._using_list_reference():
            self.set_list(data)
        return item
###### 16 ######
URL: https://github.com/stfc/PSyclone/pull/413#discussion_r296794376

Review:
Again, the docstring needs to be updated and should the return below return None rather than an empty list?
Old Version:
    @property
    def else_body(self):
        ''' Return children of the Schedule executed when the IfBlock
        evaluates to False.

        :return: Statements to be executed when IfBlock evaluates to False.
        :rtype: list of :py:class:`psyclone.psyGen.Node`
        '''
        if len(self._children) == 3:
            return self._children[2]
        return []
New Version:
    @property
    def else_body(self):
        ''' If available return the Schedule executed when the IfBlock
        evaluates to False, otherwise return None.

        :return: Schedule to be executed when IfBlock evaluates \
            to False, if it doesn't exist returns None.
        :rtype: :py:class:`psyclone.psyGen.Schedule` or NoneType
        '''
        if len(self._children) == 3:
            return self._children[2]
        return None
###### 17 ######
URL: https://github.com/metoppv/improver/pull/827/files/b76730a7c991d4dc8726e80e524c41b652ce4d58#r270913988

Review:
I'm a fan of brevity, but this seems a little extreme. Is there nothing interesting to say about process at all? If not, I guess a full stop would be something at least 😄
Old Version:
    def process(self, cubes_in):
        """
        Concatenate cubes

        Args:
            cubes_in (iris.cube.CubeList):
                Cube or list of cubes to be concatenated

        Returns:
            result (iris.cube.Cube):
                Cube concatenated along master coord

        Raises:
            ValueError:
                If master coordinate is not present on all "cubes_in"
        """
        # create copies of input cubes so as not to modify in place
        if isinstance(cubes_in, iris.cube.Cube):
            cubes = iris.cube.CubeList([cubes_in.copy()])
        else:
            cubes = iris.cube.CubeList([])
            for cube in cubes_in:
                cubes.append(cube.copy())

        # check master coordinate is on cubes - if not, throw error
        if not all(cube.coords(self.master_coord) for cube in cubes):
            raise ValueError(
                "Master coordinate {} is not present on input cube(s)".format(
                    self.master_coord))

        # slice over requested coordinates
        for coord_to_slice_over in self.coords_to_slice_over:
            cubes = self._slice_over_coordinate(cubes, coord_to_slice_over)

        # remove unmatched attributes
        equalise_cube_attributes(cubes, silent=self.silent_attributes)

        # remove cube variable names
        strip_var_names(cubes)

        # promote scalar coordinates to auxiliary as necessary
        associated_master_cubelist = iris.cube.CubeList([])
        for cube in cubes:
            associated_master_cubelist.append(
                self._associate_any_coordinate_with_master_coordinate(cube))

        # concatenate cube
        result = associated_master_cubelist.concatenate_cube()
        return result
New Version:
    def process(self, cubes_in):
        """
        Processes a list of cubes to ensure compatibility before calling the
        iris.cube.CubeList.concatenate_cube() method. Removes mismatched
        attributes, strips var_names from the cube and coordinates, and slices
        over any requested dimensions to avoid coordinate mismatch errors (eg
        for concatenating cubes with differently numbered realizations).

        Args:
            cubes_in (iris.cube.CubeList):
                Cube or list of cubes to be concatenated

        Returns:
            result (iris.cube.Cube):
                Cube concatenated along master coord

        Raises:
            ValueError:
                If master coordinate is not present on all "cubes_in"
        """
        # create copies of input cubes so as not to modify in place
        if isinstance(cubes_in, iris.cube.Cube):
            cubes = iris.cube.CubeList([cubes_in.copy()])
        else:
            cubes = iris.cube.CubeList([])
            for cube in cubes_in:
                cubes.append(cube.copy())

        # check master coordinate is on cubes - if not, throw error
        if not all(cube.coords(self.master_coord) for cube in cubes):
            raise ValueError(
                "Master coordinate {} is not present on input cube(s)".format(
                    self.master_coord))

        # slice over requested coordinates
        for coord_to_slice_over in self.coords_to_slice_over:
            cubes = self._slice_over_coordinate(cubes, coord_to_slice_over)

        # remove unmatched attributes
        equalise_cube_attributes(cubes, silent=self.silent_attributes)

        # remove cube variable names
        strip_var_names(cubes)

        # promote scalar coordinates to auxiliary as necessary
        associated_master_cubelist = iris.cube.CubeList([])
        for cube in cubes:
            associated_master_cubelist.append(
                self._associate_any_coordinate_with_master_coordinate(cube))

        # concatenate cube
        result = associated_master_cubelist.concatenate_cube()
        return result
###### 18 ######
URL: https://github.com/alteryx/evalml/pull/682#discussion_r412524712

Review:
Oh, so I had asked you to update the docstring to talk about model family. But I see we're keying on `pipeline_name`, which will be unique to each pipeline, even though we could theoretically define multiple pipelines per model family. This is fine, but let's change the comment back. Sorry, I misunderstood at first! Suggested:\r\n\r\n> Returns a pandas.DataFrame with scoring results from the highest-scoring set of parameters used with each pipeline\r\n\r\nAlso do we have a convention\u00a0yet for using \"\\`\" in docstrings, like for `pandas.DataFrame`?
Old Version:
    @property
    def rankings(self):
        """Returns a pandas.DataFrame with scoring results from the best pipeline from each model family"""
        return self.full_rankings.drop_duplicates(subset="pipeline_name", keep="first")
New Version:
    @property
    def rankings(self):
        """Returns a pandas.DataFrame with scoring results from the highest-scoring set of parameters used with each pipeline."""
        return self.full_rankings.drop_duplicates(subset="pipeline_name", keep="first")
###### 19 ######
URL: https://github.com/terrapower/armi/pull/1171/files/8c1f5fa737f5051c6d6e9613d9ecbefad72d17e0#r1102186697

Review: This docstring needs to be updated as well.

Old Version:
    def getComponentArea(self, cold=False):
        r"""Computes the area for the hexagon with n number of circular holes in cm^2."""
        od = self.getDimension("od", cold=cold)
        holeOP = self.getDimension("holeOP", cold=cold)
        mult = self.getDimension("mult")
        hexArea = math.sqrt(3.0) / 2.0 * (holeOP ** 2)
        circularArea = math.pi * ((od / 2.0) ** 2)
        area = mult * (circularArea - hexArea)
        return area

New Version:
    def getComponentArea(self, cold=False):
        r"""Computes the area for the circle with one hexagonal hole."""
        od = self.getDimension("od", cold=cold)
        holeOP = self.getDimension("holeOP", cold=cold)
        mult = self.getDimension("mult")
        hexArea = math.sqrt(3.0) / 2.0 * (holeOP ** 2)
        circularArea = math.pi * ((od / 2.0) ** 2)
        area = mult * (circularArea - hexArea)
        return area

###### 20 ######
URL: https://github.com/Project-MONAI/MONAI/pull/4289/files#r877253536

Review:
for 3D it's volume...not sure if the name could be improved

Old Version:
def box_area(boxes: NdarrayOrTensor) -> NdarrayOrTensor:
    """
    This function computes the area of each box
    Args:
        boxes: bounding box, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
    Returns:
        area of boxes, with size of (N,).
    """

    if not check_boxes(boxes):
        raise ValueError("Given boxes has invalid values. The box size must be non-negative.")

    spatial_dims = get_spatial_dims(boxes=boxes)

    area = boxes[:, spatial_dims] - boxes[:, 0] + TO_REMOVE
    for axis in range(1, spatial_dims):
        area = area * (boxes[:, axis + spatial_dims] - boxes[:, axis] + TO_REMOVE)

    # convert numpy to tensor if needed
    area_t, *_ = convert_data_type(area, torch.Tensor)

    # check if NaN or Inf, especially for half precision
    if area_t.isnan().any() or area_t.isinf().any():
        if area_t.dtype is torch.float16:
            raise ValueError("Box area is NaN or Inf. boxes is float16. Please change to float32 and test it again.")
        else:
            raise ValueError("Box area is NaN or Inf.")

    # convert tensor back to numpy if needed
    area, *_ = convert_to_dst_type(src=area_t, dst=area)
    return area
New Version:
def box_area(boxes: NdarrayOrTensor) -> NdarrayOrTensor:
    """
    This function computes the area (2D) or volume (3D) of each box.
    Half precision is not recommended for this function as it may cause overflow, especially for 3D images.
    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
    Returns:
        area (2D) or volume (3D) of boxes, with size of (N,).
    Example:
        .. code-block:: python
            boxes = torch.ones(10,6)
            # we do computation with torch.float32 to avoid overflow
            compute_dtype = torch.float32
            area = box_area(boxes=boxes.to(dtype=compute_dtype))  # torch.float32, size of (10,)
    """

    if not is_valid_box_values(boxes):
        raise ValueError("Given boxes has invalid values. The box size must be non-negative.")

    spatial_dims = get_spatial_dims(boxes=boxes)

    area = boxes[:, spatial_dims] - boxes[:, 0] + TO_REMOVE
    for axis in range(1, spatial_dims):
        area = area * (boxes[:, axis + spatial_dims] - boxes[:, axis] + TO_REMOVE)

    # convert numpy to tensor if needed
    area_t, *_ = convert_data_type(area, torch.Tensor)

    # check if NaN or Inf, especially for half precision
    if area_t.isnan().any() or area_t.isinf().any():
        if area_t.dtype is torch.float16:
            raise ValueError("Box area is NaN or Inf. boxes is float16. Please change to float32 and test it again.")
        else:
            raise ValueError("Box area is NaN or Inf.")

    return area

###### 21 ######
URL: https://github.com/NREL/reV/pull/72/commits/4d8592400706971c7e59644982bb75b588c01354#r351407399

Review:
Just for one site. I realize i forgot to update some of the docstrings. Fixed.


Old Version:
    @staticmethod
    def _get_site_mem_req(shape, dtype, n=100):
        """Get the memory requirement to collect a dataset of shape and dtype
        Parameters
        ----------
        shape : tuple
            Shape of dataset to be collected (n_time, n_sites)
        dtype : np.dtype
            Numpy dtype of dataset (disk dtype)
        n : int
            Number of sites to prototype the memory req with.
        Returns
        -------
        site_mem : float
            Memory requirement for the full dataset shape and dtype in bytes.
        """

        site_mem = sys.getsizeof(np.ones((shape[0], n), dtype=dtype)) / n
        return site_mem

New Version:
    @staticmethod
    def _get_site_mem_req(shape, dtype, n=100):
        """Get the memory requirement to collect one site from a dataset of
        shape and dtype
        Parameters
        ----------
        shape : tuple
            Shape of dataset to be collected (n_time, n_sites)
        dtype : np.dtype
            Numpy dtype of dataset (disk dtype)
        n : int
            Number of sites to prototype the memory req with.
        Returns
        -------
        site_mem : float
            Memory requirement in bytes for one site from a dataset with
            shape and dtype.
        """

        site_mem = sys.getsizeof(np.ones((shape[0], n), dtype=dtype)) / n
return site_mem

###### 22 ######
URL: https://github.com/pyRiemann/pyRiemann/pull/111#discussion_r635571167

Review: ok then update docstring

Old Version:
def normalize(X, norm):
    """Normalize a set of matrices, using trace or determinant.
    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of matrices, at least 2D ndarray. Matrices must be square for
        trace-normalization, and invertible for determinant-normalization.
    norm : {"trace", "determinant"}
        The type of normalization.
    Returns
    -------
    Xn : ndarray, shape (..., n, n)
        The set of normalized matrices, same dimensions as X.
    """
    if X.ndim < 2:
        raise ValueError('Input must have at least 2 dimensions')
    if X.shape[-2] != X.shape[-1]:
        raise ValueError('Matrices must be square')

    if norm == "trace":
        num = numpy.trace(X, axis1=-2, axis2=-1)
    elif norm  == "determinant":
        num = numpy.abs(numpy.linalg.det(X)) ** (1 / X.shape[-1])
    else:
        raise ValueError("'%s' is not a supported normalization" % norm)

    while num.ndim != X.ndim:
        num = num[..., numpy.newaxis]
    Xn = numpy.divide(X, num)
    return Xn

New Version:
def normalize(X, norm):
    """Normalize a set of square matrices, using trace or determinant.
    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray. Matrices must be
        invertible for determinant-normalization.
    norm : {"trace", "determinant"}
        The type of normalization.
    Returns
    -------
    Xn : ndarray, shape (..., n, n)
        The set of normalized matrices, same dimensions as X.
    """
    if X.ndim < 2:
        raise ValueError('Input must have at least 2 dimensions')
    if X.shape[-2] != X.shape[-1]:
        raise ValueError('Matrices must be square')

    if norm == "trace":
        denom = numpy.trace(X, axis1=-2, axis2=-1)
    elif norm  == "determinant":
        denom = numpy.abs(numpy.linalg.det(X)) ** (1 / X.shape[-1])
    else:
        raise ValueError("'%s' is not a supported normalization" % norm)

    while denom.ndim != X.ndim:
        denom = denom[..., numpy.newaxis]
    Xn = X / denom
    return Xn

###### 23 ######
URL: https://github.com/artefactual/archivematica-storage-service/pull/33#discussion_r18488678

Review: 
Let's make sure to document this, and/or raise - I could see this causing some confusing bugs.


Old Version:
    def _create_local_directory(self, path, mode=None):
        """
        Creates a local directory at 'path' with 'mode' (default 775). 
        """
        if mode is None:
            mode = (stat.S_IRUSR + stat.S_IWUSR + stat.S_IXUSR +
                    stat.S_IRGRP + stat.S_IWGRP + stat.S_IXGRP +
                    stat.S_IROTH +                stat.S_IXOTH)
        dir_path = os.path.dirname(path)
        if not dir_path:
            return
        try:
            os.makedirs(dir_path, mode)
        except os.error as e:
            # If the leaf node already exists, that's fine
            if e.errno != errno.EEXIST:
                LOGGER.warning("Could not create storage directory: %s", e)
                raise

        # os.makedirs may ignore the mode when creating directories, so force
        # the permissions here. Some spaces (eg CIFS) doesn't allow chmod, so
        # wrap it in a try-catch and ignore the failure.
        try:
            os.chmod(os.path.dirname(path), mode)
        except os.error as e:
            LOGGER.warning(e)

New Version:
    def _create_local_directory(self, path, mode=None):
        """
        Creates directory structure for `path` with `mode` (default 775).
        :param path: path to create the directories for.  Should end with a / or
            a filename, or final directory may not be created. If path is empty,
            no directories are created.
        :param mode: (optional) Permissions to create the directories with
            represented in octal (like bash or the stat module)
        """
        if mode is None:
            mode = (stat.S_IRUSR + stat.S_IWUSR + stat.S_IXUSR +
                    stat.S_IRGRP + stat.S_IWGRP + stat.S_IXGRP +
                    stat.S_IROTH +                stat.S_IXOTH)
        dir_path = os.path.dirname(path)
        if not dir_path:
            return
        try:
            os.makedirs(dir_path, mode)
        except os.error as e:
            # If the leaf node already exists, that's fine
            if e.errno != errno.EEXIST:
                LOGGER.warning("Could not create storage directory: %s", e)
                raise

        # os.makedirs may ignore the mode when creating directories, so force
        # the permissions here. Some spaces (eg CIFS) doesn't allow chmod, so
        # wrap it in a try-catch and ignore the failure.
        try:
            os.chmod(os.path.dirname(path), mode)
        except os.error as e:
            LOGGER.warning(e)

###### 24 ######
URL: https://github.com/kensho-technologies/graphql-compiler/pull/187#discussion_r255645481

Review:
FIx the docstring, there's no \"three parts of a macro edge\"
Old Version:
def get_and_validate_macro_edge_info(schema, ast, macro_directives, macro_edge_args,
                                     type_equivalence_hints=None):
    """Return a tuple of ASTs with the three parts of a macro edge given the directive mapping.
    Args:
        schema: GraphQL schema object, created using the GraphQL library
        ast: GraphQL library AST OperationDefinition object, describing the GraphQL that is defining
             the macro edge.
        macro_directives: Dict[str, List[Tuple[AST object, Directive]]], mapping the name of an
                          encountered directive to a list of its appearances, each described by
                          a tuple containing the AST with that directive and the directive object
                          itself.
        macro_edge_args: dict mapping strings to any type, containing any arguments the macro edge
                         requires in order to function.
        type_equivalence_hints: optional dict of GraphQL interface or type -> GraphQL union.
                                Used as a workaround for GraphQL's lack of support for
                                inheritance across "types" (i.e. non-interfaces), as well as a
                                workaround for Gremlin's total lack of inheritance-awareness.
                                The key-value pairs in the dict specify that the "key" type
                                is equivalent to the "value" type, i.e. that the GraphQL type or
                                interface in the key is the most-derived common supertype
                                of every GraphQL type in the "value" GraphQL union.
                                Recursive expansion of type equivalence hints is not performed,
                                and only type-level correctness of this argument is enforced.
                                See README.md for more details on everything this parameter does.
                                *****
                                Be very careful with this option, as bad input here will
                                lead to incorrect output queries being generated.
                                *****
    Returns:
        tuple (class name for macro, name of macro edge, MacroEdgeDescriptor),
        where the first two values are strings and the last one is a MacroEdgeDescriptor object
    """
    _validate_macro_ast_with_macro_directives(schema, ast, macro_directives)

    macro_defn_ast, macro_defn_directive = macro_directives[MacroEdgeDefinitionDirective.name][0]
    macro_target_ast, _ = macro_directives[MacroEdgeTargetDirective.name][0]

    # TODO(predrag): Required further validation:
    # - the macro definition directive AST contains only @filter/@fold directives together with
    #   the target directive;
    # - after adding an output, the macro compiles successfully, the macro args and necessary and
    #   sufficient for the macro, and the macro args' types match the inferred types of the
    #   runtime parameters in the macro.

    class_ast = get_only_selection_from_ast(ast)
    class_name = get_ast_field_name(class_ast)

    _validate_class_selection_ast(class_ast, macro_defn_ast)

    macro_edge_name = macro_defn_directive.arguments['name'].value

    _validate_macro_edge_name_for_class_name(schema, class_name, macro_edge_name)

    _make_macro_edge_descriptor()

return class_name, macro_edge_name

New Version:
def get_and_validate_macro_edge_info(schema, ast, macro_directives, macro_edge_args,
                                     type_equivalence_hints=None):
    """Return a tuple with the three parts of information that uniquely describe a macro edge.
    Args:
        schema: GraphQL schema object, created using the GraphQL library
        ast: GraphQL library AST OperationDefinition object, describing the GraphQL that is defining
             the macro edge.
        macro_directives: Dict[str, List[Tuple[AST object, Directive]]], mapping the name of an
                          encountered directive to a list of its appearances, each described by
                          a tuple containing the AST with that directive and the directive object
                          itself.
        macro_edge_args: dict mapping strings to any type, containing any arguments the macro edge
                         requires in order to function.
        type_equivalence_hints: optional dict of GraphQL interface or type -> GraphQL union.
                                Used as a workaround for GraphQL's lack of support for
                                inheritance across "types" (i.e. non-interfaces), as well as a
                                workaround for Gremlin's total lack of inheritance-awareness.
                                The key-value pairs in the dict specify that the "key" type
                                is equivalent to the "value" type, i.e. that the GraphQL type or
                                interface in the key is the most-derived common supertype
                                of every GraphQL type in the "value" GraphQL union.
                                Recursive expansion of type equivalence hints is not performed,
                                and only type-level correctness of this argument is enforced.
                                See README.md for more details on everything this parameter does.
                                *****
                                Be very careful with this option, as bad input here will
                                lead to incorrect output queries being generated.
                                *****
    Returns:
        tuple (class name for macro, name of macro edge, MacroEdgeDescriptor),
        where the first two values are strings and the last one is a MacroEdgeDescriptor object
    """
    _validate_macro_ast_with_macro_directives(schema, ast, macro_directives)

    macro_defn_ast, macro_defn_directive = macro_directives[MacroEdgeDefinitionDirective.name][0]
    # macro_target_ast, _ = macro_directives[MacroEdgeTargetDirective.name][0]

    # TODO(predrag): Required further validation:
    # - the macro definition directive AST contains only @filter/@fold directives together with
    #   the target directive;
    # - after adding an output, the macro compiles successfully, the macro args and necessary and
    #   sufficient for the macro, and the macro args' types match the inferred types of the
    #   runtime parameters in the macro.
    class_ast = get_only_selection_from_ast(ast)
    class_name = get_ast_field_name(class_ast)
    _validate_class_selection_ast(class_ast, macro_defn_ast)
    macro_edge_name = macro_defn_directive.arguments['name'].value
    _validate_macro_edge_name_for_class_name(schema, class_name, macro_edge_name)
    _make_macro_edge_descriptor()
return class_name, macro_edge_name

###### 25 ######
URL: https://github.com/proteneer/timemachine/pull/640#discussion_r808454189

Review: 
Nit: updates behavior but does not update docstring

Old Version:
def compute_or_load_bond_smirks_matches(mol, smirks_list):
    """Return an array of ordered bonds and an array of their assigned types
    Notes
    -----
    * Uses OpenEye for substructure searches
    * Order within smirks_list matters
        "First match wins."
        For example, if bond (a,b) can be matched by smirks_list[2], smirks_list[5], ..., assign type 2
    * Order within each smirks pattern matters
        For example, "[#6:1]~[#1:2]" and "[#1:1]~[#6:2]" will match atom pairs in the opposite order
    """
    if not mol.HasProp(BOND_SMIRK_MATCH_CACHE):
        oemol = convert_to_oe(mol)
        AromaticityModel.assign(oemol)

        bond_idxs = []  # [B, 2]
        type_idxs = []  # [B]

        for type_idx, smirks in enumerate(smirks_list):
            matches = oe_match_smirks(smirks, oemol)

            for matched_indices in matches:
                a, b = matched_indices[0], matched_indices[1]
                forward_matched_bond = [a, b]
                reverse_matched_bond = [b, a]

                already_assigned = forward_matched_bond in bond_idxs or reverse_matched_bond in bond_idxs

                if not already_assigned:
                    bond_idxs.append(forward_matched_bond)
                    type_idxs.append(type_idx)
        mol.SetProp(BOND_SMIRK_MATCH_CACHE, base64.b64encode(pickle.dumps((bond_idxs, type_idxs))))
    else:
        bond_idxs, type_idxs = pickle.loads(base64.b64decode(mol.GetProp(BOND_SMIRK_MATCH_CACHE)))
    return np.array(bond_idxs), np.array(type_idxs)

New Version:
def compute_or_load_bond_smirks_matches(mol, smirks_list):
    """Unless already cached in mol's "BondSmirkMatchCache" property, uses OpenEye to compute arrays of ordered bonds and their assigned types.
    Notes
    -----
    * Uses OpenEye for substructure searches
    * Order within smirks_list matters
        "First match wins."
        For example, if bond (a,b) can be matched by smirks_list[2], smirks_list[5], ..., assign type 2
    * Order within each smirks pattern matters
        For example, "[#6:1]~[#1:2]" and "[#1:1]~[#6:2]" will match atom pairs in the opposite order
    """
    if not mol.HasProp(BOND_SMIRK_MATCH_CACHE):
        oemol = convert_to_oe(mol)
        AromaticityModel.assign(oemol)

        bond_idxs = []  # [B, 2]
        type_idxs = []  # [B]

        for type_idx, smirks in enumerate(smirks_list):
            matches = oe_match_smirks(smirks, oemol)

            for matched_indices in matches:
                a, b = matched_indices[0], matched_indices[1]
                forward_matched_bond = [a, b]
                reverse_matched_bond = [b, a]

                already_assigned = forward_matched_bond in bond_idxs or reverse_matched_bond in bond_idxs

                if not already_assigned:
                    bond_idxs.append(forward_matched_bond)
                    type_idxs.append(type_idx)
        mol.SetProp(BOND_SMIRK_MATCH_CACHE, base64.b64encode(pickle.dumps((bond_idxs, type_idxs))))
    else:
        bond_idxs, type_idxs = pickle.loads(base64.b64decode(mol.GetProp(BOND_SMIRK_MATCH_CACHE)))
    return np.array(bond_idxs), np.array(type_idxs)