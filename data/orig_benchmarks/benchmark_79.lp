d0(o5).
d4(o5).
d2(o5).
d3(o0).
d2(o2).
d1(o3).

d2(X) :- d1(X), d4(X).
d4(X) :- d3(X), d1(X).
d5(X) :- d0(X), d3(X).
d0(X) :- d5(X), d4(X).
d2(X) :- d3(X), d4(X).
d1(X) :- d0(X), d2(X).
d3(X) :- d0(X), d1(X).
d3(X) :- d0(X), d2(X).
d3(X) :- d0(X), d1(X).
d5(X) :- d3(X), d1(X).
d4(X) :- d0(X), d1(X).