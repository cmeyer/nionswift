#### TODO
- [ ] rewrite evaluator to be threaded (async) so it can report progress and be paused/resumed
- [ ] implementations should register point by point function, figure out naming
- [ ] allow computations to return data and scalars (e.g., alignment offset)
- [ ] generalize map-rgb, i.e. n (three) scalar functions mapped to colors (rgb)
- [ ] generalize execute/commit to not require access to a computation object
- [ ] expand inputs to allow more flexibility, i.e. measured (average from range), literal (index), alternate literal (fractional index)
- [ ] expand inputs also to allow graphical inputs (range or index) but optional
- [ ] literal inputs should allow calibrated inputs
- [ ] add option outputs, e.g. shifts in align-zlp
- [ ] map sequence of eels spectra with interval should show scalar within interval, (mask or selection?)
- [ ] generalize scalar maps to allow user to choose mask or selection
- [ ] consider mapping to allow element reductions (average, min, max, stddev), element expansions (histogram), collection aggregations (sum, average, rolling average, counts, min, max)

#### Describing Computations

The `ComputationProcessor` describes a computation that can be executed on data. It contains: title, sources, parameters, attributes, out regions, outputs. Attributes are things like connections between its sources/outputs, i.e. for graphical connections.

It also provides a method `needs_update_for_event` for sources to indicate whether they should be considered stale given specific data update events.

`register_processing_descriptions` allows packages to register a computation processor constructed from a dict. See also `register_processors`. See also `Symbolic._processors`.

#### Executing the Computation

The computation is executed directly as a script or as an object of type `ComputationHandlerLike`.

The computation is executed in two phases: execute and commit. The execute phase performs the computation, and the commit phase applies the results to the data store. The execute phase may be threaded and must not access the project objects directly, while the commit phase is not threaded and accesses the project objects directly.

`register_computation_type` associates the processing id with a `ComputationHandlerLike`, which currently has two methods: execute and commit.

TODO: The progress interface should include the ability to pause a computation, pause after finishing current frame, resume the computation, and restart with current frame.

TODO: Execute should be able to track an acquisition as the frames come in, i.e. a pipeline.

#### ProcessingComputation

ProcessingComputation is a ComputationHandlerLike which takes a subclass of ProcessingBase to evaluate and commit.

ProcessingBase is an initial attempt to make a point-bv-point processor that can be mapped to a collection in the ProcessingComputation.

#### Notes

Some ComputationHandlerLike objects registered with register_computation_type contain label, inputs, outputs, parameters, attributes, etc. that should be moved to the ComputationProcessor.

Other ComputationHandlerLike objects do not contain that information, but supply that information when creating the computation object. This should be migrated to the ComputationProcessor.

How can register_processing_descriptions / ComputationProcessor represent computations that take 1d/2d arrays but are suitable for point-bv-point processing?

ComputationProcessorRequirement.is_data_item_valid should take a DataMetadata, not a DataItem, so that it can be used to validate data in a collection.

"IterableComputationHandler" should have a start, process, and end; begin should return a state object that is passed to process and end. The handler should mark itself as threadable and should not keep state between calls to process. The state object may have specific requirements, e.g. be immutable after begin, if threaded.

Anything that can be applied to collection can be applied live.

There is a difference between applying a computation the first time vs maintaining the connection. The user interface can make it easy for the user to configure the common case, i.e. 'align ZLP using selected index' as setup, but allow user to configure a measured input, literal input, etc.

A computation can have multiple outputs, e.g. aligned data and shifts. Outputs may come in a variety of forms, e.g. data items, scalars, metadata, regions of interest, etc. Some of the outputs may be variable, i.e. find particles in the data.

Scalar outputs (measurements? values?) should reachable from the data item they were computed from. Maybe through the computation? `r01.thickness_map.thickness_value`?

Scalar outputs might also be derivable through common operators: e.g. min, max, average, stddev, histogram, etc. `average(r01)`.

Processing descriptions could also describe the options for parameter inputs, e.g. measured from range, literal index, graphical selection, etc. The align-ZLP could specify to use the ZLP position of the current sequence index as the default target index.

Align 1d could be a general procedure with a 'measurement' input of 'ZLP position'. Align 2d could be the same.

General case is to look for ways to provide common operations with variants that can be registered by speific packages. Then let the package provide menu items to "shortcut" to configure the common operation.

Measurement should be either parallelized point by point or done in a pre-pass, i.e. handle relative shifts.

#### Align ZLP project

- uses sequence index for reference if it is a simple sequence of 1d data; otherwise uses index=0 of flattened collection.
- align_zlp_xdata works on collections with 1d data; or 2d data that is not a collection.
- outputs aligned data and shifts, both having same collection dimensions as input.
- TODO: drop support for 2d data that is not a collection.
- works on single 1d spectra, navigable collections of 1d spectra, and live 1d spectra.
- "index" should be measured, absolute index, or fractional index.
- User input → range or index
- Graphical selection → range or index
- Zero loss peak position (pixel number) as measured over range → align_position
- Zero loss peak position (pixel number) as measured at index → align_position
- User value → align_position

align_eels_spectrum(data1, align_position)
  zlp_position = measure_zlp(data1)
  shift_eels_spectrum(data1, zlp_position, align_position)
