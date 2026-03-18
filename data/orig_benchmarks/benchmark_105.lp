d1(o4).
d2(o3).
d4(o5).
d3(o4).
d4(o4).
d5(o4).

d2(X) :- d4(X), d1(X), d3(X).
d1(X) :- d4(X), d2(X), d3(X).
d1(X) :- d2(X), d5(X), d3(X).
d0(X) :- d4(X), d1(X), d3(X).
d1(X) :- d2(X), d0(X), d5(X).
d5(X) :- d2(X), d3(X), d4(X).
d1(X) :- d2(X), d0(X), d3(X).
d5(X) :- d4(X), d0(X), d2(X).
d3(X) :- d1(X), d0(X), d5(X).
d0(X) :- d4(X), d1(X), d3(X).
d2(X) :- d4(X), d1(X), d3(X).