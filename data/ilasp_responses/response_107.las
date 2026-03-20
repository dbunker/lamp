%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.007s
%% Hypothesis Space Generation             : 0.187s
%% Conflict analysis                       : 0.868s
%%   - Positive Examples                   : 0.868s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.001s
%% Hypothesis Search                       : 0.389s
%% Total                                   : 1.49s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d2(V1) :- d0(V1).
d0(V1) :- d3(V1).
d0(V1) :- d4(V1).
d5(V1) :- d0(V1).
d4(V1) :- d2(V1); not d3(V1).

