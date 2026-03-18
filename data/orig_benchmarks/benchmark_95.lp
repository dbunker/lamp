d4(o0).
d3(o2).
d0(o2).
d2(o5).
d1(o4).
d2(o3).
d2(o4).
d5(o5).
d5(o4).

d5(X) :- d2(X), d3(X), d4(X).
d4(X) :- d2(X), d1(X), d5(X).
d4(X) :- d2(X), d5(X), d3(X).
d3(X) :- d4(X), d0(X), d2(X).
d1(X) :- d4(X), d0(X), d3(X).
d3(X) :- d4(X), d0(X), d2(X).
d0(X) :- d4(X), d1(X), d3(X).
d5(X) :- d2(X), d1(X), d0(X).
d1(X) :- d2(X), d0(X), d3(X).
d1(X) :- d4(X), d2(X), d3(X).
d3(X) :- d2(X), d0(X), d5(X).