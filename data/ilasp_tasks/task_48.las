#constant(obj, o0).
#constant(obj, o4).
#constant(obj, o5).
#constant(obj, o6).
#constant(obj, o7).
#constant(obj, o8).

#modeh(d1(var(obj))).
#modeh(d4(var(obj))).
#modeh(d5(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).

#pos(eg1, {
  d1(o8), d4(o0), d5(o0), d5(o8)
}, {
  d1(o4), d1(o5), d1(o7), d4(o4), d4(o5), d4(o6), d4(o7), d5(o4), d5(o7)
}, {
  d0(o7).
  d1(o0).
  d1(o6).
  d2(o0).
  d2(o8).
  d3(o4).
  d4(o8).
  d5(o5).
  d5(o6).
}).