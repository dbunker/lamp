%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.202s
%% Conflict analysis                       : 1.016s
%%   - Positive Examples                   : 1.016s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 0.341s
%% Total                                   : 1.614s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d5(V1) :- d1(V1).
d3(V1) :- d2(V1).
d0(V1) :- d4(V1).
d1(V1) :- d0(V1).
d4(V1) :- d5(V1); not d2(V1).

