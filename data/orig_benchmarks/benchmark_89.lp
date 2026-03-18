d1(o5).
d1(o8).
d3(o5).
d0(o7).
d2(o0).

d4(X) :- d2(X), d0(X).
d3(X) :- d1(X), d5(X).
d0(X) :- d4(X), d2(X).
d3(X) :- d4(X), d5(X).
d2(X) :- d1(X), d0(X).
d2(X) :- d5(X), d3(X).
d2(X) :- d1(X), d5(X).
d5(X) :- d4(X), d0(X).
d5(X) :- d4(X), d2(X).
d5(X) :- d1(X), d3(X).
d2(X) :- d4(X), d5(X).